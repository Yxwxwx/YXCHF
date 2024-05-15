pub struct Atom {
    pub atomic_number: i32,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct Mole {
    pub atoms: Vec<Atom>,
    pub n_elec: i32,
    pub n_multi: i32,
    pub basis_info: String,
}

impl Mole {
    pub fn new(gto: String, n_elec: Option<i32>, n_multi: Option<i32>, basis_info: String) -> Self {
        let mut atoms = Vec::new();
        for atom_str in gto.split(';') {
            let parts: Vec<&str> = atom_str.trim().split_whitespace().collect();
            if parts.len() != 4 {
                panic!("Invalid atom string");
            }
            let element_label = parts[0].to_string();
            let atomic_number = match element_label.as_str() {
                "H" => 1,
                "C" => 6,
                "N" => 7,
                "O" => 8,
                "F" => 9,
                "S" => 16,
                "Cl" => 17,
                _ => {
                    eprintln!(
                        "read_dotxyz: element label \"{}\" is not recognized",
                        element_label
                    );
                    panic!("Did not recognize element label in .xyz file");
                }
            };
            let x: f64 = parts[1].parse().expect("Failed to parse x coordinate");
            let y: f64 = parts[2].parse().expect("Failed to parse y coordinate");
            let z: f64 = parts[3].parse().expect("Failed to parse z coordinate");
            atoms.push(Atom { atomic_number, x, y, z });
        }
        
        let n_elec = n_elec.unwrap_or(0);
        let n_multi = n_multi.unwrap_or(0);

        Self { atoms, n_elec, n_multi, basis_info }
    }
    pub fn print(&self) {
        println!("Basis Info: {}", self.basis_info);
        println!("Number of Electrons: {}", self.n_elec);
        println!("Number of Multiplicities: {}", self.n_multi);
        println!("Atoms:");
        for atom in &self.atoms {
            println!("Atomic number: {}, x: {}, y: {}, z: {}", atom.atomic_number, atom.x, atom.y, atom.z);
        }
        println!();
    }
}


pub struct Shell {
    pub l: i32,
    pub exp: Vec<f64>,
    pub coeff: Vec<f64>,
    pub coord: Vec<f64>,
    pub ngto: i32,
    pub shl: Vec<Vec<i32>>,
}

impl Shell {
    pub fn new(l: i32, exp: Vec<f64>, coeff: Vec<f64>, coord: Vec<f64>) -> Self {
        let mut shl = Vec::new();
        match l {
            0 => shl.push(vec![0, 0, 0]),
            1 => shl.extend_from_slice(&[[1, 0, 0].to_vec(), [0, 1, 0].to_vec(), [0, 0, 1].to_vec()]),
            2 => shl.extend_from_slice(&[[1, 1, 0].to_vec(), [1, 0, 1].to_vec(), [0, 1, 1].to_vec(), [2, 0, 0].to_vec(), [0, 2, 0].to_vec(), [0, 0, 2].to_vec()]),
            _ => panic!("Do not support angular momentum l {}", l),
        }
        let ngto = (l + 1) * (l + 2) / 2;
        Self { l, exp, coeff, coord, ngto, shl }
    }
}

pub struct Shells {
    pub shells: Vec<Shell>,
}

impl Shells {
    pub fn new(mole: &Mole) -> Self {
        match mole.basis_info.as_str() {
            "sto-3g" => Self::make_sto3g_basis(mole),
            _ => panic!("Do not support basis {}", mole.basis_info),
        }
    }
    pub fn print(&self) {
        for shell in &self.shells {
            println!("Shell: l = {}, exp = {:?}, coeff = {:?}, coord = {:?}", shell.l, shell.exp, shell.coeff, shell.coord);
        }
        println!();
    }
    fn make_sto3g_basis(mole: &Mole) -> Self {
        let mut shells = Vec::new();

        for atom in &mole.atoms {
            match atom.atomic_number {
                1 => {
                    shells.push(Shell::new(
                        0,
                        vec![3.425250910, 0.623913730, 0.168855400], 
                        vec![0.15432897, 0.53532814, 0.44463454], 
                        vec![atom.x, atom.y, atom.z],
                    ));
                }
                6 => {
                    shells.push(Shell::new(
                        0,
                        vec![71.616837000, 13.045096000, 3.530512200],
                        vec![0.15432897, 0.53532814, 0.44463454],
                        vec![atom.x, atom.y, atom.z],
                    ));
                    shells.push(Shell::new(
                        0,
                        vec![2.941249400, 0.683483100, 0.222289900],
                        vec![-0.09996723, 0.39951283, 0.70011547],
                        vec![atom.x, atom.y, atom.z],
                    ));
                    shells.push(Shell::new(
                        1,
                        vec![2.941249400, 0.683483100, 0.222289900],
                        vec![0.15591627, 0.60768372, 0.39195739],
                        vec![atom.x, atom.y, atom.z],
                    ));
                }
                8 => {
                    shells.push(Shell::new(
                        0,
                        vec![130.709320000, 23.808861000, 6.443608300],
                        vec![0.15432897, 0.53532814, 0.44463454],
                        vec![atom.x, atom.y, atom.z],
                    ));
                    shells.push(Shell::new(
                        0,
                        vec![5.033151300, 1.169596100, 0.380389000],
                        vec![-0.09996723, 0.39951283, 0.70011547],
                        vec![atom.x, atom.y, atom.z],
                    ));
                    shells.push(Shell::new(
                        1,
                        vec![5.033151300, 1.169596100, 0.380389000],
                        vec![0.15591627, 0.60768372, 0.39195739],
                        vec![atom.x, atom.y, atom.z],
                    ));
                }
                _ => panic!("Do not know STO-3G basis for this Z"),
            }
        }

        Self { shells }
    }
}
