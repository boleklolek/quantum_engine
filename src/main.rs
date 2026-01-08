use std::env;
use std::fs::File;
use std::io::Read;

use nalgebra::DMatrix;

use quantum_engine::input::parser::Input;
use quantum_engine::system::molecule::Molecule;
use quantum_engine::basis::reader::load_basis;
use quantum_engine::scf::scf_cycle::run_scf;
use quantum_engine::gradients::total::compute_gradients;
use quantum_engine::hessian::total::compute_hessian;
use quantum_engine::vibrations::frequencies::compute_frequencies;

fn main() {
    // -------------------------------------------------
    // 1. Parse CLI
    // -------------------------------------------------
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage:");
        eprintln!("  quantum_engine input.yaml geometry.xyz");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let xyz_file   = &args[2];

    // -------------------------------------------------
    // 2. Read input file
    // -------------------------------------------------
    let mut input_text = String::new();
    File::open(input_file)
        .expect("Cannot open input file")
        .read_to_string(&mut input_text)
        .expect("Cannot read input file");

    let input = Input::from_yaml(&input_text)
        .expect("Invalid input file");

    // -------------------------------------------------
    // 3. Read XYZ → Molecule
    // -------------------------------------------------
    let molecule =
        Molecule::from_xyz(xyz_file)
            .expect("Invalid XYZ file");

    println!("Molecule loaded: {} atoms", molecule.atoms.len());

    // -------------------------------------------------
    // 4. Load basis
    // -------------------------------------------------
    let (shells, shell_centers) =
        load_basis(
            &molecule,
            &input.basis,
        ).expect("Basis load failed");

    println!("AO basis size: {}", shells.last().unwrap().offset
                                      + shells.last().unwrap().orbitals.len());

    // -------------------------------------------------
    // 5. SCF
    // -------------------------------------------------
    println!("Running SCF...");
    let scf_result =
        run_scf(
            &molecule,
            &shells,
            &shell_centers,
            &input.scf,
            &input.dft,
        );

    println!("SCF converged in {} iterations",
             scf_result.iterations);
    println!("Total energy: {:.10} Eh",
             scf_result.energy);

    // -------------------------------------------------
    // 6. Gradients (optional)
    // -------------------------------------------------
    if input.task.compute_gradients {
        println!("Computing gradients...");
        let grad =
            compute_gradients(
                &molecule,
                &shells,
                &shell_centers,
                &scf_result,
            );

        grad.print();
    }

    // -------------------------------------------------
    // 7. Hessian + frequencies (optional)
    // -------------------------------------------------
    if input.task.compute_hessian {
        println!("Computing Hessian...");
        let hess =
            compute_hessian(
                &molecule,
                &shells,
                &shell_centers,
                &scf_result,
            );

        if input.task.compute_frequencies {
            let freqs =
                compute_frequencies(&molecule, &hess);
            freqs.print();
        }
    }

    println!("Done.");
}

