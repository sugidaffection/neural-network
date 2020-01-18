extern crate rand;

use rand::{thread_rng, Rng};

struct Matrix {
    col: usize,
    row: usize,
    array: Vec<Vec<f64>>
}

impl Matrix {
    fn new(col: usize, row: usize) -> Matrix {
        let array = vec![vec![0.0;col];row];

        Matrix {
            col: col,
            row: row,
            array: array
        }
    }

    fn random_matrix(&mut self) {
        self.array = self.array.iter_mut().map(|x| {
            x.iter_mut().map(|_| thread_rng().gen_range(0.0,1.0)).collect()
        }).collect::<Vec<Vec<f64>>>();
    }

    fn multiply(&mut self, mat: &mut Matrix) -> Option<Matrix> {
        if self.col == mat.row {
            let mut array:Vec<Vec<f64>> = vec![vec![0.0;self.col];self.row];
            for row in 0..self.row {
                for col in 0..self.col {
                    array[row][col] = self.array[row][col] * mat.get_array()[col][0];
                }
            }

            array = array.iter_mut().map(|x| vec![x.iter().sum()]).collect();
            let mut mat = Matrix::new(mat.col, self.row);
            mat.set_array(&mut array);
            return Some(mat);
        }
        None
    }

    fn add(&mut self, mat: &mut Matrix) -> Option<Matrix> {
        if self.row == mat.row {
            let mut array = self.array.iter().zip(mat.get_array().iter()).map(|(x,y)| x.iter().zip(y.iter()).map(|(v,z)| v+z).collect()).collect();
            let mut mat = Matrix::new(mat.col, self.row);
                    mat.set_array(&mut array);
            return Some(mat);
        }

        None
    }

    fn set_array(&mut self, array: &mut Vec<Vec<f64>>) -> &mut Matrix {
        self.array = array.clone();
        self
    }

    fn get_array(&mut self) -> Vec<Vec<f64>> {
        self.array.clone()
    }

    fn sigmoid(&mut self) -> &mut Matrix {
        self.array = self.array.iter_mut().map(|x| {
            x.iter_mut().map(|v| {
                v.exp() / (v.exp() + 1.0)
            }).collect()
        }).collect();

        self
    }

    fn display(&mut self) -> &mut Matrix {
        self.array.iter().for_each(|x| {
            println!("{:?}", x);
        });

        self
    }
}

#[allow(dead_code)]
struct NeuralNetwork {
    input: usize,
    hidden: usize,
    output: usize,
    wh: Matrix,
    wo: Matrix,
    bh: Matrix,
    bo: Matrix
}

impl NeuralNetwork {
    fn new(input: usize, hidden: usize, output: usize) -> NeuralNetwork {
        let mut wh = Matrix::new(input, hidden);
            wh.random_matrix();
        let mut wo = Matrix::new(hidden, output);
            wo.random_matrix();
        let mut bh = Matrix::new(1, hidden);
            bh.random_matrix();
        let mut bo = Matrix::new(1, output);
            bo.random_matrix();

        NeuralNetwork {
            input: input,
            hidden: hidden,
            output: output,
            wh: wh,
            wo: wo,
            bh: bh,
            bo: bo
        }
    }

    fn feed_forward(&mut self, input: &mut Matrix) -> Matrix {
        let mut hidden = self.wh.multiply(input).unwrap();
                hidden = hidden.add(&mut self.bh).unwrap();
                hidden.sigmoid();

        hidden.display();

        let mut output = self.wo.multiply(&mut hidden).unwrap();
                output = output.add(&mut self.bo).unwrap();
                output.sigmoid();

        output
    }
}

fn main() {
    let mut nn = NeuralNetwork::new(2, 2, 1);
    let mut input = Matrix::new(1, 2);
            input.set_array(&mut vec![vec![1.0],vec![0.0]]);

    println!("INPUT : ");
            input.display();
    println!("HIDDEN : ");
    let mut output = nn.feed_forward(&mut input);
    println!("OUTPUT : ");
            output.display();
    
}
