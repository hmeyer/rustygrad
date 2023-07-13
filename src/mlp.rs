use crate::{Module, Dense, Value};
use std::fmt;


/// A simple Multi-Layer-Perceptron.
pub struct Mlp {
    l: Vec<Dense>,
}

impl Mlp {
    /// Create a Multi-Layer-Perceptron.
    pub fn new(n_inputs: usize, n_outputs: &[usize]) -> Mlp {
        let mut l = Vec::new();
        for i in 0..n_outputs.len() {
            let nin = if i == 0 {n_inputs} else {n_outputs[i - 1]};
            let nout = n_outputs[i];
            let activation = i != n_outputs.len() - 1;
            l.push(Dense::new(nin, nout, activation));
        }
        Mlp { l }
    }
    /// Call the MLP using an input.
    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut x = self.l[0].call(x);
        for l in self.l.iter().skip(1) {
            x = l.call(&x);
        }
        x
    }
}

impl Module for Mlp {
   fn parameters(&self) -> Vec<Value> {
        self.l.iter().fold(Vec::new(), |mut acc, x| {acc.append(&mut x.parameters()); acc})
    }
}

impl fmt::Display for Mlp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mlp of [{}]", self.l.iter().map(|l| format!("{}", l)).collect::<Vec<_>>().join(", "))
    }
}
