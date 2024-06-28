#include "Mole.h"

int main(int argc, char* argv[]) {
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    Mole mole;
    auto start = std::chrono::high_resolution_clock::now();
    auto atoms = mole.read_geometry(filename);
    auto shells = mole.make_sto3g_basis(atoms);
    auto end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> diff = end-start;
    std::cout << "initialize molecule cost: " << diff.count() << " s\n";

    const int ndocc = mole.m_nelec / 2;
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    try
    {
        // compute the nuclear repulsion energy
        auto enuc = 0.0;
        for (size_t i = 0; i < atoms.size(); i++) 
            for (size_t j = i + 1; j < atoms.size(); j++) {
                auto xij = atoms[i].x - atoms[j].x;
                auto yij = atoms[i].y - atoms[j].y;
                auto zij = atoms[i].z - atoms[j].z;
                auto r2 = xij * xij + yij * yij + zij * zij;
                auto r = sqrt(r2);
                enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
            }
        std::cout << "\tNuclear repulsion energy = " << enuc << std::endl;

        mole.m_S = mole.calc_ovlp(shells);
        std::cout << "\n\tOverlap Integrals:\n";
        //std::cout << mole.m_S << std::endl;

        mole.m_H = mole.calc_core_h(shells, atoms);
        std::cout << "\n\tCore Hamiltonian:\n";
        //std::cout << mole.m_H << std::endl;

        Matrix D;
        //use core Hamiltonian eigenstates to guess density
        Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(mole.m_H, mole.m_S);
        auto eps = gen_eig_solver.eigenvalues();
        auto C = gen_eig_solver.eigenvectors();

        //std::cout << "\n\tInitial C Matrix:\n";
        //std::cout << C << std::endl;

        // compute density, D = C(occ) . C(occ)T
        auto C_occ = C.leftCols(ndocc);
        D = C_occ * C_occ.transpose();

        //std::cout << "\n\tInitial Density Matrix:\n";
        //std::cout << D << std::endl;

        const auto maxiter = 100;
        const double conv = 1e-10;
        auto iter = 0;
        double rmsd = 0.0;
        double ediff = 0.0;
        double ehf = 0.0;
        std::vector<Mole::DIISInfo> diis_info;

        do
        {

          const auto tstart = std::chrono::high_resolution_clock::now();
          ++iter;

          // Save a copy of the energy and the density          
          auto ehf_last = ehf;
          auto D_last = D;
          auto F = mole.m_H;
          F += mole.calc_eri_direct(shells, D);        

          auto diis_r = mole.compute_diis_r(F, D, mole.m_S);
          mole.save_diis_info(iter, ehf, F, D, diis_r, diis_info, 8);

          // compute HF energy
          ehf = 0.0;
          for (auto i = 0; i < nao; i++)
            for (auto j = 0; j < nao; j++) ehf += D(i, j) * (mole.m_H(i, j) + F(i, j));
          
          // compute difference with last iteration
          ediff = ehf - ehf_last;
          rmsd = 0.5 * std::sqrt((diis_r.array() * diis_r.array()).mean());;

          if (fabs(rmsd) < 0.1 || iter > 8)
          { 
            F = mole.cdiis_minimize(diis_info);
            Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, mole.m_S);
            auto eps = gen_eig_solver.eigenvalues();
            auto C = gen_eig_solver.eigenvectors();
            // compute density, D = C(occ) . C(occ)T
            auto C_occ = C.leftCols(ndocc);
            D = C_occ * C_occ.transpose();
          }  
          else {
            Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, mole.m_S);
            auto eps = gen_eig_solver.eigenvalues();
            auto C = gen_eig_solver.eigenvectors();
    
            // compute density, D = C(occ) . C(occ)T
            auto C_occ = C.leftCols(ndocc);
            D = C_occ * C_occ.transpose();
          }

          const auto tstop = std::chrono::high_resolution_clock::now();
          const std::chrono::duration<double> time_elapsed = tstop - tstart;

          if (iter == 1) 
            std::cout << "\n\n Iter        E(elec)              E(tot)             "
                     "  Delta(E)             RMS(D)         Time(s)\n";
          

          printf(" %02d %20.12f %20.12f %20.12f %20.12f %10.5lf\n", iter, ehf,
             ehf + enuc, ediff, rmsd, time_elapsed.count());


        } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));
        
        printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);
    }

    catch (const char* ex) {
        std::cerr << "caught exception: " << ex << std::endl;
    return 1;
    } catch (std::string& ex) {
        std::cerr << "caught exception: " << ex << std::endl;
    return 1;
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    return 1;
    } catch (...) {
        std::cerr << "caught unknown exception\n";
    return 1;
    }
    
    return 0;
}

Matrix Mole::calc_ovlp(const std::vector<Shell>& shells) {

    int nshls = shells.size();
    auto sum_shls = nshls * (nshls + 1) / 2;

    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Eigen::IOFormat fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    Matrix S = Matrix::Zero(nao, nao);

    int  ijkl[2];

    int* ik = new int[sum_shls], * jl = new int[sum_shls];
    for (int i = 0, ij=0; i < nshls; ++i)
        for (int j = i; j < nshls; ++j, ++ij)
        {
            ik[ij] = i;
            jl[ij] = j;
        }
    
    int ipr, jpr;
    int ij, di, dj, x, y;
    int fi, fj, fij;
    double* buf;
    //Matrix buf;
 #pragma omp parallel default(none) \
             shared(nshls, sum_shls, shells, ik, jl, S) \
             private(ipr, jpr, ij, di, x, dj, y, ijkl, fi, fj,fij, buf) 
 #pragma omp for nowait schedule(dynamic, 2)

    for (ij = 0; ij < sum_shls; ij++)
    {
        ipr = ik[ij];
        jpr = jl[ij];

        di = shells[ipr].ngto;
        x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }

        dj = shells[jpr].ngto;
        y = 0;
        for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
        }

        ijkl[0] = ipr;
        ijkl[1] = jpr;

        buf = new double[di * dj]();
        int1e_ovlp_cart(buf, shells, ijkl);
        //buf.setZero(di, dj);
        //int1e_ovlp_cart(buf, shells, ijkl);
        //std::cout << "\n" << buf << std::endl;
        fij = 0;
        for (fi = 0; fi < di; ++fi) {
            for (fj = 0; fj < dj; ++fj, ++fij) {
                S(x + fi, y + fj) = buf[fij];
                S(y + fj, x + fi) = buf[fij];
            }
        }
        delete[] buf;
    }

    delete[]ik;
    delete[]jl;  
    return S;
}

Matrix Mole::calc_kin(const std::vector<Shell>& shells) {

    int nshls = shells.size();
    auto sum_shls = nshls * (nshls + 1) / 2;

    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Eigen::IOFormat fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    Matrix T = Matrix::Zero(nao, nao);

    int  ijkl[2];

    int* ik = new int[sum_shls], * jl = new int[sum_shls];
    for (int i = 0, ij=0; i < nshls; ++i)
        for (int j = i; j < nshls; ++j, ++ij)
        {
            ik[ij] = i;
            jl[ij] = j;
        }

    int ipr, jpr;
    int ij, di, dj, x, y;
    int fi, fj, fij;
    double* buf;
    //Matrix buf;
 #pragma omp parallel default(none) \
             shared(nshls, sum_shls, shells, ik, jl, T) \
             private(ipr, jpr, ij, di, x, dj, y, ijkl, fi, fj,fij, buf) 
 #pragma omp for nowait schedule(dynamic, 2)

    for (ij = 0; ij < sum_shls; ij++)
    {
        ipr = ik[ij];
        jpr = jl[ij];

        di = shells[ipr].ngto;
        x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }

        dj = shells[jpr].ngto;
        y = 0;
        for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
        }

        ijkl[0] = ipr;
        ijkl[1] = jpr;

        buf = new double[di * dj]();
        int1e_kin_cart(buf, shells, ijkl);
        //buf.setZero(di, dj);
        //int1e_ovlp_cart(buf, shells, ijkl);
        //std::cout << "\n" << buf << std::endl;
        fij = 0;
        for (fi = 0; fi < di; ++fi) {
            for (fj = 0; fj < dj; ++fj, ++fij) {
                T(x + fi, y + fj) = buf[fij];
                T(y + fj, x + fi) = buf[fij];
            }
        }
        delete[] buf;
    }

    delete[]ik;
    delete[]jl;  
    return T;
}


Matrix Mole::calc_nuc(const std::vector<Shell>& shells, const std::vector<Atom>& atoms) {

    std::vector<std::pair<int, std::vector<double>>> q;
    for (const auto& atom : atoms) {
         q.push_back({static_cast<int>(atom.atomic_number),
                   {{atom.x, atom.y, atom.z}}});
    }

    int nshls = shells.size();
    auto sum_shls = nshls * (nshls + 1) / 2;

    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Eigen::IOFormat fmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
    Matrix V = Matrix::Zero(nao, nao);

    int  ijkl[2];

    int* ik = new int[sum_shls], * jl = new int[sum_shls];
    for (int i = 0, ij=0; i < nshls; ++i)
        for (int j = i; j < nshls; ++j, ++ij)
        {
            ik[ij] = i;
            jl[ij] = j;
        }

    int ipr, jpr;
    int ij, di, dj, x, y;
    int fi, fj, fij;
    double* buf;
    //Matrix buf;
 #pragma omp parallel default(none) \
             shared(nshls, sum_shls, shells, atoms, ik, jl, V, q) \
             private(ipr, jpr, ij, di, x, dj, y, ijkl, fi, fj,fij, buf) 
 #pragma omp for nowait schedule(dynamic, 2)

    for (ij = 0; ij < sum_shls; ij++)
    {
        ipr = ik[ij];
        jpr = jl[ij];

        di = shells[ipr].ngto;
        x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }

        dj = shells[jpr].ngto;
        y = 0;
        for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
        }

        ijkl[0] = ipr;
        ijkl[1] = jpr;

        buf = new double[di * dj]();
        int1e_nuc_cart(buf, shells, ijkl, q);
        //buf.setZero(di, dj);
        //int1e_ovlp_cart(buf, shells, ijkl);
        //std::cout << "\n" << buf << std::endl;
        fij = 0;
        for (fi = 0; fi < di; ++fi) {
            for (fj = 0; fj < dj; ++fj, ++fij) {
                V(x + fi, y + fj) = buf[fij];
                V(y + fj, x + fi) = buf[fij];
            }
        }
        delete[] buf;
    }

    delete[]ik;
    delete[]jl;  
    
    return V;
}

Tensor Mole::calc_eri(const std::vector<Shell>& shells) {

    int nshls = shells.size();
    auto sum_shls = nshls * (nshls + 1) / 2;

    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Tensor I(nao, nao, nao, nao);

    int* ik = new int[nshls * nshls], * jl = new int[nshls * nshls];
    for (int i = 0, ij = 0; i < nshls; i++) {
        for (int j = i; j < nshls; j++, ij++) {
            ik[ij] = i;
            jl[ij] = j;
        }
    }

    int ipr, jpr, kpr, lpr;
    int ij, kl, di, dj, dk, dl, x, y, z, w, ijkl[4];
    int sum_kl;
    int fi, fj, fk, fl, fijkl;
    double* buf;
#pragma omp parallel default(none) \
             shared(nshls, sum_shls, shells, ik, jl, I) \
             private(ipr, jpr, kpr, lpr, sum_kl, ij, kl, di, x, dj, y, dk, z, dl, w, ijkl, fi, fj, fk, fl, fijkl, buf) 
#pragma omp for nowait schedule(dynamic, 2)   
    for (ij = 0; ij < sum_shls; ij++) {
        ipr = ik[ij];
        jpr = jl[ij];

        di = shells[ipr].ngto;
        x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }

        dj = shells[jpr].ngto;
        y = 0;
        for (auto j = 0; j < jpr; j++) {
            y += shells[j].ngto;
        }
        ijkl[0] = ipr;
        ijkl[1] = jpr;

        sum_kl = ipr * (ipr + 1) / 2;
        for (kl = sum_kl; kl < sum_shls; kl++) {
            kpr = ik[kl];
            lpr = jl[kl];

            dk = shells[kpr].ngto;
            z = 0;
            for (int k = 0; k < kpr; k++)
            {
                z += shells[k].ngto;
            }

            dl = shells[lpr].ngto;
            w = 0;
            for (int l = 0; l < lpr; l++)
            {
                w += shells[l].ngto;
            }

            ijkl[2] = kpr;
            ijkl[3] = lpr;

            buf = new double[di*dj*dk*dl]();
            int2e_cart(buf, shells, ijkl);
            fijkl = 0;
            for ( fi = 0; fi < di; fi++) {
                for (fj = 0; fj < dj; fj++) {
                    for (fk = 0; fk < dk; fk++) {
                        for (fl = 0; fl < dl; fl++, fijkl++) {
                            I(x+fi, y+fj, z+fk, w+fl) = buf[fijkl];
                            I(y+fj, x+fi, z+fk, w+fl) = buf[fijkl];
                            I(x+fi, y+fj, w+fl, z+fk) = buf[fijkl];
                            I(y+fj, x+fi, w+fl, z+fk) = buf[fijkl];
                            I(z+fk, w+fl, x+fi, y+fj) = buf[fijkl];
                            I(w+fl, z+fk, x+fi, y+fj) = buf[fijkl];
                            I(z+fk, w+fl, y+fj, x+fi) = buf[fijkl];
                            I(w+fl, z+fk, y+fj, x+fi) = buf[fijkl];
                        }
                    } 
                }  
            }
            delete[] buf;
            
        }  
    }
    
    delete[]ik;
    delete[]jl;
    return I;
}

Matrix Mole::calc_eri_direct(const std::vector<Shell>& shells, const Matrix& D) {

    int nshls = shells.size();

    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Matrix G = Matrix::Zero(nao, nao);

    int s1, s2, s3, s4, s4_max;
    int di, dj, dk, dl, x, y, z, w, ijkl[4];
    int fi, fj, fk, fl, fijkl;
    double* buf;
    double s1234_deg;

#pragma omp parallel default(none) \
             shared(nshls, shells, G, D) \
             private(s1, s2, s3, s4, s4_max, di, x, dj, y, dk, z, dl, w, ijkl, fi, fj, fk, fl, fijkl, buf, s1234_deg) 
#pragma omp for nowait schedule(dynamic, 2)   
    for (s1 = 0; s1 < nshls; s1++) {
        di = shells[s1].ngto;
        x = 0;
        for (int i = 0; i < s1; i++) {
            x += shells[i].ngto;
        }

        for (s2 = 0; s2 <= s1; s2++) {
            dj = shells[s2].ngto;
            y = 0;
            for (auto j = 0; j < s2; j++) {
                y += shells[j].ngto;
            }


            ijkl[0] = s1;
            ijkl[1] = s2;

            for (s3 = 0; s3 <= s1; s3++) {
                dk = shells[s3].ngto;
                z  = 0;
                for (int k = 0; k < s3; k++)
                {   
                    z += shells[k].ngto;
                }   

                s4_max = (s1 == s3) ? s2 : s3;
                for (s4 = 0; s4 <= s4_max; s4++) {
                    dl = shells[s4].ngto;
                    w = 0;
                    for (int l = 0; l < s4; l++)
                    {
                        w += shells[l].ngto;
                    }

                    
                    ijkl[2] = s3;
                    ijkl[3] = s4;
                    s1234_deg = degeneracy(ijkl);

                    buf = new double[di*dj*dk*dl]();
                    int2e_cart(buf, shells, ijkl);
                    fijkl = 0;
                    for ( fi = 0; fi < di; fi++) {
                        for (fj = 0; fj < dj; fj++) {
                            for (fk = 0; fk < dk; fk++) {
                                for (fl = 0; fl < dl; fl++, fijkl++) {
                                    G(x + fi, y + fj) += D(z + fk, w + fl) * buf[fijkl] * s1234_deg;
                                    G(z + fk, w + fl) += D(x + fi, y + fj) * buf[fijkl] * s1234_deg;
                                    G(x + fi, z + fk) -= 0.25 * D(y + fj, w + fl) * buf[fijkl] * s1234_deg;
                                    G(y + fj, w + fl) -= 0.25 * D(x + fi, z + fk) * buf[fijkl] * s1234_deg;
                                    G(x + fi, w + fl) -= 0.25 * D(y + fj, z + fk) * buf[fijkl] * s1234_deg;
                                    G(y + fj, z + fk) -= 0.25 * D(x + fi, w + fl) * buf[fijkl] * s1234_deg;
                                }
                            } 
                        }  
                    }
                    delete[] buf;
                }      
            } 
        }
    }

    Matrix Gt = G.transpose();
    return 0.5 * (G + Gt);
}

Matrix Mole::compute_diis_r(const Matrix& F, const Matrix& D, const Matrix& S) {
    return F * D * S - S * D * F;
}
void Mole::save_diis_info(int iter, double E_elec0, Matrix& F, Matrix& D, Matrix& DIIS_R, std::vector<DIISInfo>& diis_info, int n_diis) {
    DIISInfo info;
    info.scf_iter = iter;
    info.energy = E_elec0;
    info.fock_matrix = F;
    info.density_matrix = D;
    info.diis_error = DIIS_R;

    diis_info.push_back(info);
    if (diis_info.size() > n_diis) {
        diis_info.erase(diis_info.begin());
    }
}

Matrix Mole::cdiis_minimize(std::vector<DIISInfo>& diis_info)
    {
    auto nx = diis_info.size();
    std::vector<Matrix> fock_matrices;
    std::vector<Matrix> diis_resid;

    for (const DIISInfo& info : diis_info)
    {
        const Matrix fock_matrix = info.fock_matrix;
        fock_matrices.push_back(fock_matrix);
    }
    for (const DIISInfo& info : diis_info)
    {
        const Matrix diis_errors = info.diis_error;
        diis_resid.push_back(diis_errors);
    }

    auto dim_B = nx + 1;
    Matrix m_B = Matrix::Zero(dim_B, dim_B);

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            m_B(i, j) = (diis_resid[i].array() * diis_resid[j].array()).sum();
        }

    }
    for (int i = 0; i < dim_B - 1; i++) {
        m_B(i, dim_B - 1) = -1;
        m_B(dim_B - 1, i) = -1;
    }
    m_B(dim_B - 1, dim_B - 1) = 0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dim_B);
    rhs(dim_B - 1) = -1;
    Eigen::VectorXd coeff = m_B.fullPivLu().solve(rhs);

    int m_nao = fock_matrices[0].rows();
    Matrix F_DIIS = Matrix::Zero(m_nao, m_nao);

    for (int x = 0; x < coeff.size() - 1; x++)
    {
        F_DIIS += coeff(x) * fock_matrices[x];
    }
    //std::cout << F_DIIS << std::endl;
    return F_DIIS;
}