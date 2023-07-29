# minetest-interpretabilty-notebook

Jupyter notebook for the interpretablity section of the minetester blog post

## Jono's refactor

What used to be

```python
class Network(nn.Module):
    @nn.compact
    def __call__(self, x, l1_inp=False, l2_inp=False, l3_inp=False, inp=None):
        l1, l2, l3 = None, None, None

        if not (l1_inp or l2_inp or l3_inp):
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x / (255.0)
            x = nn.Conv(
                32,
                kernel_size=(8, 8),
                strides=(4, 4),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            l1 = x
        if l1_inp:
            x = inp
        if not (l2_inp or l3_inp):
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            l2 = x
        if l2_inp:
            x = inp
        if not (l3_inp):
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            l3 = x
        if l3_inp:
            x = inp
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        return x, (l1, l2, l3)


def network_gradient(output_fn, target="input"):
    if target == "input":

        def f(x):
            hidden, layers = network.apply(network_params, x)
            action_logits = actor.apply(actor_params, hidden)
            critic_values = critic.apply(critic_params, hidden)
            output = output_fn(action_logits, critic_values, hidden, layers)
            return output

    if target == "l1":

        def f(x):
            hidden, layers = network.apply(network_params, None, l1_inp=True, inp=x)
            action_logits = actor.apply(actor_params, hidden)
            critic_values = critic.apply(critic_params, hidden)
            output = output_fn(action_logits, critic_values, hidden, layers)
            return output

    if target == "l2":

        def f(x):
            hidden, layers = network.apply(network_params, None, l2_inp=True, inp=x)
            action_logits = actor.apply(actor_params, hidden)
            critic_values = critic.apply(critic_params, hidden)
            output = output_fn(action_logits, critic_values, hidden, layers)
            return output

    if target == "l3":

        def f(x):
            hidden, layers = network.apply(network_params, None, l3_inp=True, inp=x)
            action_logits = actor.apply(actor_params, hidden)
            critic_values = critic.apply(critic_params, hidden)
            output = output_fn(action_logits, critic_values, hidden, layers)
            return output

    if target == "network":

        def f():
            def f(x):
                hidden, layers = x, (None, None, None)
                action_logits = actor.apply(actor_params, hidden)
                critic_values = critic.apply(critic_params, hidden)
                output = output_fn(action_logits, critic_values, hidden, layers)

            return output

    return jax.value_and_grad(f)


def deep_dream(
    init_input, output_fn, target="input", lr=1e3, n_iter=600, clip_low=0, clip_high=255
):
    hidden, layers = network.apply(network_params, init_input)
    if target == "input":
        x = init_input
    if target == "l1":
        x = layers[0]
    if target == "l2":
        x = layers[1]
    if target == "l3":
        x = layers[2]
    if target == "network":
        x = hidden

    f = jax.jit(network_gradient(output_fn, target=target))

    for i in range(n_iter):
        value, grad = f(x)
        x += lr * grad
        x = jnp.clip(x, clip_low, clip_high)
        if i % 200 == 0:
            print("Iteration:", i, "Value", value)

    return x
```

now is

```python
class Network(nn.Module):
    @nn.compact
    def __call__(self, x, start_at=0):
        if start_at < 1:
            x = jnp.transpose(x, (0, 2, 3, 1))
            x = x / (255.0)
            x = nn.Conv(
                32,
                kernel_size=(8, 8),
                strides=(4, 4),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv0", x)

        if start_at < 2:
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv1", x)

        if start_at < 3:
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            self.sow("intermediates", "activations", x)
            self.perturb("conv2", x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x
        )
        x = nn.relu(x)
        return x


def actor_critic(x, intem_activs, output_fn):
    action_logits = actor.apply(actor_params, x)
    critic_values = critic.apply(critic_params, x)
    output = output_fn(action_logits, critic_values, x, intem_activs)
    return output


def deep_dream(
    init_input, output_fn, target=0, lr=1e3, n_iter=600, clip_low=0, clip_high=255
):
    hidden, intem_activs = network.apply(
        network_params, init_input, mutable="intermediates"
    )
    intem_activs = intem_activs["intermediates"]["activations"]
    variables = dict(network_params)
    variables["perturbations"] = perturbations

    if target > len(intem_activs):

        def forward(x):
            return actor_critic(x, None, output_fn)

        x = hidden

    else:

        def forward(x):
            x, intermediates = network.apply(
                variables, x, start_at=target, mutable="intermediates"
            )
            return actor_critic(
                x, intermediates["intermediates"]["activations"], output_fn
            )

        if target == 0:
            x = init_input
        else:
            x = intem_activs[target - 1]

    f = jax.jit(jax.value_and_grad(forward))

    for i in range(n_iter):
        value, grad = f(x)
        x += lr * grad
        x = jnp.clip(x, clip_low, clip_high)
        if i % 200 == 0:
            print("Iteration:", i, "Value", value)

    return x
```

And some minor other calls because what used to be

```python
hidden, intermediate_activations = network.apply(network_params, x)
```

Now is

```python
hidden, intermediates = network.apply(network_params, x, mutable="intermediates")
intermediate_activations = intermediates["intermediates"]["activations"]
```
