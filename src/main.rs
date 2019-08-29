use std::iter::Iterator;
use std::cmp::{Ord, Ordering};
use std::io::{BufReader, BufRead, BufWriter, Write, stdin, stdout};
use std::fs::File;
use std::ops::Add;
use std::error::Error;
use crate::ErrorType::{FileIOError, ParseError};
use rand::Rng;
use rand::prelude::ThreadRng;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && "train".eq(&args[1]) {
        train();
    } else {
        test(if args.len() > 3 { 1 } else { -1 });
    }
}

fn test(player_turn: i8) {
    let mut learner = Learner::new(0.5, 0.95);
    learner.load("qTable").unwrap_or(());
    println!("your mark is {}", if player_turn == 1 { 'o' } else { 'x' });
    let mut game_state = new_game();
    let mut turn = 1;
    let mut rng = rand::thread_rng();
    loop {
        let mut action: usize;
        if turn == player_turn {
            print!("your turn\naction=>");
            stdout().flush().unwrap();
            let mut s = String::new();
            stdin().read_line(&mut s);
            action = s.trim().parse().unwrap();
        } else {
            action = learner.select_action(to_state_code(game_state, -player_turn), 0f64, &mut rng);
            print!("COM turn\naction=>{}\n", action);
        }
        let (new_state, next_turn, award, end_game) = update_state(game_state, turn, action);
        game_state = new_state;
        turn = next_turn;
        let x: Vec<char> = game_state.iter().map(|&v| if v == 1 { 'o' } else if v == -1 { 'x' } else { ' ' }).collect();
        println!("---\n{}{}{}\n{}{}{}\n{}{}{}\n---", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]);
        if end_game {
            println!("game end.");
            println!("{}", if award == 0.0 { "draw." } else if turn * player_turn > 0 { "you lose." } else { "you win." });
            break;
        }
    }
}

fn train() {
    let mut learner = Learner::new(0.5, 0.95);
    learner.load("qTable").unwrap_or(());
    loop {
        for i in 0..10_000 {
            let mut game_state = new_game();
            let mut turn = 1;
            let mut rng = rand::thread_rng();
            loop {
                let s: String = game_state.iter().map(|&v| { if v == 1 { 'o' } else if v == -1 { 'x' } else { ' ' } }).collect();
                print!("{}", s);
                let action = learner.select_action(to_state_code(game_state, turn), 0.3, &mut rng);
                let (new_state, next_turn, award, end_game) = update_state(game_state, turn, action);
                let a = learner.select_action(to_state_code(new_state, -turn), 0.0, &mut rng);
                let (next_state, _, _, _) = update_state(new_state, -turn, a);
                learner.update(to_state_code(game_state, turn), action, to_state_code(next_state, turn), award);
                turn = next_turn;
                game_state = new_state;
                let s: String = game_state.iter().map(|&v| { if v == 1 { 'o' } else if v == -1 { 'x' } else { ' ' } }).collect();
                println!("=>{}", s);
                if end_game {
                    break;
                }
            }
        }
        learner.store("qTable");
    }
}


fn print_q_table(learner: Learner) {
    for i in 0..learner.table.len() {
        let state = from_state_code(i, 1);
        let s: String = state.iter().map(|&v| { if v == 1 { 'o' } else if v == -1 { 'x' } else { ' ' } }).collect();
        println!("[{}]:{:?}", s, learner.table[i]);
    }
}

type GameState = [i8; 9];

fn new_game() -> GameState {
    [0; 9]
}

fn to_state_code(state: GameState, turn: i8) -> usize {
    let list: Vec<usize> = state
        .iter()
        .map(|&v| { if v == turn { 1 } else if v == -turn { 2 } else { 0 } })
        .collect();
    let mut sum = 0;
    let mut bias = 1;
    for i in list {
        sum += bias * i;
        bias *= 3;
    }
    sum
}

fn from_state_code(mut state: usize, turn: i8) -> GameState {
    let mut game_state: GameState = [0; 9];
    for i in 0..9 {
        game_state[8 - i] = match state % 3 {
            0 => 0,
            1 => turn,
            2 => -turn,
            _ => panic!()
        };
        state /= 3;
    }
    game_state
}

fn update_state(current_state: GameState, turn: i8, action: usize) -> (GameState, i8, f64, bool) {
    if current_state[action] != 0 { return (current_state, turn, -1f64, false); }
    let mut new_state = current_state;
    new_state[action] = turn;
    for i in 0..3 {
        let sum: i8 = [new_state[3 * i], new_state[3 * i + 1], new_state[3 * i + 2]].iter().sum();
        if sum == turn * 3 {
            return (new_state, -turn, 1f64, true);
        }
    }
    for i in 0..3 {
        let sum: i8 = [new_state[i], new_state[3 + i], new_state[6 + i]].iter().sum();
        if sum == turn * 3 {
            return (new_state, -turn, 1f64, true);
        }
    }
    let sum: i8 = [new_state[0], new_state[4], new_state[8]].iter().sum();
    if sum == turn * 3 {
        return (new_state, -turn, 1f64, true);
    }
    let sum: i8 = [new_state[2], new_state[4], new_state[6]].iter().sum();
    if sum == turn * 3 {
        return (new_state, -turn, 1f64, true);
    }
    let sum: i8 = new_state.iter().map(|v| v * v).sum();
    if sum >= 9 {
        return (new_state, -turn, 0f64, true);
    }
    (new_state, -turn, 0f64, false)
}

enum ErrorType {
    FileIOError,
    ParseError,
}

struct Learner {
    pub table: [[f64; 9]; 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3],
    alpha: f64,
    gamma: f64,
}

impl Learner {
    fn new(alpha: f64, gamma: f64) -> Learner {
        Learner {
            table: [[0f64; 9]; 19_683],
            alpha,
            gamma,
        }
    }

    fn q_values(&self, state: usize) -> &[f64] {
        &self.table[state][..]
    }

    fn select_action(&self, state: usize, epsilon: f64, r: &mut ThreadRng) -> usize {
        let v: f64 = r.gen();
        if v < epsilon {
            let action: f64 = r.gen();
            (action * 9f64) as usize
        } else {
            let q = self.q_values(state);
            let mut max = vec![0];
            for i in 0..q.len() {
                if q[i] == q[max[0]] {
                    max.push(i);
                } else if q[i] > q[max[0]] {
                    max.clear();
                    max.push(i);
                }
            }
            let action: f64 = r.gen();
            max[(action * max.len() as f64) as usize]
        }
    }

    fn update(&mut self, current_state: usize, action: usize, after_state: usize, award: f64) {
        let after_max = {
            let iter = self.q_values(after_state);
            let mut max = iter[0];
            for &f in iter { max = f64::max(max, f); }
            max
        };
        self.table[current_state][action] =
            (1f64 - self.alpha) * self.table[current_state][action] +
                self.alpha * (award + self.gamma * after_max);
    }

    fn load(&mut self, file_name: &str) -> Result<(), ErrorType> {
        let file = BufReader::new(match File::open(file_name) {
            Err(e) => { return Err(FileIOError); }
            Ok(file) => file
        });
        let mut state = 0;
        for line in file.lines() {
            if line.is_err() { break; }
            let line = match line {
                Err(e) => {
                    self.table = [[0f64; 9]; 19683];
                    return Err(ParseError);
                }
                Ok(v) => v
            };
            let mut action = 0;
            for i in line.split(",") {
                self.table[state][action] = match i.parse() {
                    Err(e) => {
                        self.table = [[0f64; 9]; 19683];
                        return Err(ParseError);
                    }
                    Ok(v) => v
                };
                action += 1;
            }
            state += 1;
        }
        Ok(())
    }

    fn store(&self, file_name: &str) -> Result<(), ErrorType> {
        let mut file = BufWriter::new(match File::create(file_name) {
            Err(e) => { return Err(FileIOError); }
            Ok(v) => v
        });
        for state in 0..self.table.len() {
            for action in 0..9 {
                match write!(file, "{}", self.table[state][action]) {
                    Err(e) => { return Err(FileIOError); }
                    Ok(v) => v
                };
                if action != 8 {
                    match write!(file, ",") {
                        Err(e) => { return Err(FileIOError); }
                        Ok(v) => v
                    };
                } else {
                    match writeln!(file) {
                        Err(e) => { return Err(FileIOError); }
                        Ok(v) => v
                    };
                }
            }
        }
        Ok(())
    }
}