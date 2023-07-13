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

use std::fmt;
use std::ops;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;
use rand::prelude::*;
use std::iter::zip;
use std::iter::Iterator;

// The Ops a Value supports internally.
#[derive(Debug)]
enum ValueOp {
    Variable,  // A standard leaf node value. Either a constant or a trainable variable.
    Add,  // Addition.
    Mul,  // Multiplication.
    Pow,  // Raising a value to a power.
    Relu, // Relu activation.
}

/// A basic value.
#[derive(Clone)]
pub struct Value {
    v: Rc<RefCell<ValueImpl>>,
}

struct ValueImpl {
    pub data: f64,
    pub grad: f64,
    pub op: ValueOp,
    pub children: Vec<Value>,
    pub base_and_exponent: Option<(f64, f64)>,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let v = self.v.borrow();
        let mut base_and_exponent = "".to_string();
        if let Some((b, e)) = v.base_and_exponent {
            base_and_exponent = format!("base_and_exponent={}^{} ", b, e);
        }
        write!(f, "Value(data={}, grad={}, {:?}, {}[", v.data, v.grad, v.op, base_and_exponent)?;
        for c in &v.children {
            write!(f, "{}, ", c)?;
        }
        write!(f, "])")
    }
}

impl Value {
    fn new(data: f64, op: ValueOp, children: Vec<Value>) -> Value {
        Value{v:Rc::new(RefCell::new(ValueImpl{data, grad: 0.0, op, children, base_and_exponent: None}))}
    }
    /// Access to the data.
    pub fn data(&self) -> f64 {
        self.v.borrow().data
    }
    /// Set the value.
    pub fn set_data(&mut self, data: f64) {
        self.v.borrow_mut().data = data;
    }
    /// Access to the gradient. The gradient will be zero after initialization and only be populated if `backward()` is called somewhere on the expression tree.
    pub fn grad(&self) -> f64 {
        self.v.borrow().grad
    }
    /// Raise a value to a power.
    pub fn pow(&self, exponent: f64) -> Value {
        let data = self.data();
        let r  = Value::new(data.powf(exponent), ValueOp::Pow, vec![self.clone()]);
        r.v.borrow_mut().base_and_exponent = Some((data, exponent));
        r
    }
    /// Relu activation of a value.
    pub fn relu(&self) -> Value {
        let relu_data = self.v.borrow().data.max(0.0);
        Value::new(relu_data, ValueOp::Relu, vec![self.clone()])
    }
    fn backward_impl(&mut self) {
            let mut v = self.v.borrow_mut();
        let grad = v.grad;
        match &v.op {
            ValueOp::Variable => {},
            ValueOp::Add => {
                v.children[0].add_grad(grad);
                v.children[1].add_grad(grad);
            },
            ValueOp::Mul => {
                let c0_data = v.children[0].v.borrow().data;
                let c1_data = v.children[1].v.borrow().data;
                v.children[0].add_grad(grad * c1_data);
                v.children[1].add_grad(grad * c0_data);
            },
            ValueOp::Pow => {
                let (base, exponent) = v.base_and_exponent.unwrap();
                v.children[0].add_grad(exponent * base.powf(exponent - 1.0) * grad);
            }
            ValueOp::Relu => {
                let grad = if v.data == 0.0 {0.0} else {grad};
                v.children[0].add_grad(grad);
            }
        }
    }
    // Helper for topological sort.
    fn build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<*const ValueImpl>) {
        let self_r: &ValueImpl = &self.v.borrow();
        let self_p = self_r as *const ValueImpl;
        if visited.contains(&self_p) { return; }
        visited.insert(self_p);
        for c in &self.v.borrow().children {
            c.build_topo(topo, visited);
        }
        topo.push(self.clone());
    }
    /// Run gradient backpropagation.
    pub fn backward(&mut self) {
        let mut topo = Vec::new();
        {
            let mut visited = HashSet::new();
            self.build_topo(&mut topo, &mut visited);
        }
        self.v.borrow_mut().grad = 1.0;
        for mut v in topo.into_iter().rev() {
            v.backward_impl();
        }
    }
    /// Zero out the gradient.
    pub fn zero_grad(&mut self) {
        self.v.borrow_mut().grad = 0.0;
    }
    /// Add a value to the gradient.
    pub fn add_grad(&mut self, g: f64) {
        self.v.borrow_mut().grad += g;
    }
}

/// Wrap a float in a basic value.
pub fn v(x: f64) -> Value {
    Value::new(x, ValueOp::Variable, Vec::new())
}

impl ops::Add<Value> for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        Value::new(self.v.borrow().data + rhs.v.borrow().data, ValueOp::Add, vec![self.clone(), rhs.clone()])
    }
}

impl ops::Add<f64> for Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Value {
        self + v(rhs)
    }
}

impl ops::Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * (-1.0)
    }
}

impl ops::Sub<Value> for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        self + (-rhs)
    }
}

impl ops::Sub<f64> for Value {
    type Output = Value;
    fn sub(self, rhs: f64) -> Value {
        self + (-rhs)
    }
}

impl ops::Mul<&Value> for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        Value::new(self.v.borrow().data * rhs.v.borrow().data, ValueOp::Mul, vec![self.clone(), rhs.clone()])
    }
}

impl ops::Mul<Value> for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        &self * &rhs
    }
}

impl ops::Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value {
        self * v(rhs)
    }
}

impl ops::Div<Value> for Value {
    type Output = Value;
    fn div(self, rhs: Value) -> Value {
        self * rhs.pow(-1.0)    
    }
}

impl ops::Div<f64> for Value {
    type Output = Value;
    fn div(self, rhs: f64) -> Value {
        self / v(rhs)    
    }
}

/// Trait for any NN-Module.
pub trait Module {
    fn parameters(&self) -> Vec<Value>;
    fn zero_grad(&mut self) {
        for mut p in self.parameters() {
            p.zero_grad();
        }
    } 
}

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
        let r = zip(&self.w, x).fold(v(0.0), |acc, (w, x)| acc + w * x);
        if self.activation {
            r.relu()
        } else {
            r
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
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


#[cfg(test)]
mod tests {

    #[test]
    fn todo() {
    }
}
