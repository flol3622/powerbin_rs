#[derive(Debug, Clone)]
pub struct EarlyStopper {
    pub window: usize,
    pub min_iters: usize,
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub best_history: Vec<f64>,
    pub iter: usize,
    pub last: f64,
    pub under_relax: bool,
}

impl EarlyStopper {
    pub fn new(window: usize, min_iters: usize, rel_tol: f64, abs_tol: f64) -> Self {
        Self {
            window,
            min_iters,
            rel_tol,
            abs_tol,
            best_history: vec![f64::INFINITY],
            iter: 0,
            last: f64::INFINITY,
            under_relax: false,
        }
    }

    pub fn update(&mut self, value: f64) -> bool {
        self.iter += 1;
        if value > self.last {
            self.under_relax = true;
        }
        self.last = value;

        let current_best = value.min(*self.best_history.last().unwrap());
        self.best_history.push(current_best);

        if self.iter < self.min_iters.max(self.window) {
            return false;
        }

        let best_then = self.best_history[self.best_history.len() - 1 - self.window];
        let best_now = *self.best_history.last().unwrap();
        let threshold = self.abs_tol.max(self.rel_tol * best_then.abs());
        if best_then - best_now > threshold {
            return false;
        }
        true
    }
}
