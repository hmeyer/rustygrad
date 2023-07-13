pub use crate::{Value, v, Module};
use std::iter;
use std::fmt;
use rand::prelude::*;


/// A simple Neuron.
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    activation: bool,
}

impl Neuron {
    /// Create a Neuron.
    pub fn new(n_inputs: usize, activation: bool) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut w = Vec::new();
        for _ in 0..n_inputs {
            let r: f64 = rng.gen();
            w.push(v(r * 2.0 - 1.0));
        }
        Neuron {w, b: v(0.0), activation }
    }
    /// Call the Neuron with inputs.
    pub fn call(&self, x: &[Value]) -> Value {
        assert_eq!(self.w.len(), x.len());
        let r = iter::zip(&self.w, x).fold(v(0.0), |acc, (w, x)| acc + w * x);
        if self.activation {
            r.relu()
        } else {
            r
        }
    }
}

impl Module for Neuron {
    fn parameters(&mut self) -> Box<dyn Iterator<Item=&mut Value> + '_> {
        let w = self.w.iter_mut();
        let w_and_b = w.chain(iter::once(&mut self.b)); 
        Box::new(w_and_b)
    }
}


impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self.activation {
            false => "Linear",
            true => "Relu",
        };
        write!(f, "{}Neuron({})", kind, self.w.len())
    }
}

