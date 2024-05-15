mod mole;

use mole::Shells;
use mole::Mole;

fn main() {
    let gto = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587".to_string();
    let n_elec = Some(0);
    let n_multi = Some(0);
    let basis_info = "sto-3g".to_string();
    let mole = Mole::new(gto.clone(), n_elec, n_multi, basis_info);
    mole.print();

    let shells = Shells::new(&mole);
    shells.print();

    println!("Hello World!");
}

