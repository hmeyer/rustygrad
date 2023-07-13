use crate::{Module, Neuron, Value};
use std::fmt;


/// A Dense Layer.
pub struct Dense {
    n: Vec<Neuron>,
}

impl Dense {
    /// Create a Dense Layer.
    pub fn new(n_inputs: usize, n_outputs: usize, activation: bool) -> Dense {
        let mut n = Vec::new();
        for _ in 0..n_outputs {
            n.push(Neuron::new(n_inputs, activation));
        }
        Dense { n }
    }
    /// Call the Dense Layer with an input.
    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        self.n.iter().map(|n| n.call(x)).collect()
    }
}
impl Module for Dense {
    fn parameters(&self) -> Vec<Value> {
        self.n.iter().fold(Vec::new(), |mut acc, x| {acc.append(&mut x.parameters()); acc})
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dense Layer of [{}]", self.n.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(", "))
    }
}

