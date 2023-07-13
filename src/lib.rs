/*!
 Minimal Autograd Engine in Rust.

Based on <https://github.com/karpathy/micrograd>

But in Rust.

```rust
use rustygrad::v;

let x = v(0.0);
let mut y = (x.clone() * - 3.0 - 3.0).pow(2.0).relu();
y.backward();
// y = relu((-3 * x - 3)^2)
// y = relu(9x^2 + 18x +9)
// dy/dx = drelu(18x + 18)
// dy/dx(0) = 18
assert_eq!(x.grad(), 18.0);
```

*/

mod engine;
pub use crate::engine::{Value, v};

mod module;
pub use crate::module::Module;

mod neuron;
pub use crate::neuron::Neuron;

mod dense;
pub use crate::dense::Dense;

mod mlp;
pub use crate::mlp::Mlp;
