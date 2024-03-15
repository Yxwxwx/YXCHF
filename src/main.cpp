#include "Mole.h"

int main(int argc, char* argv[]) {
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    Mole mole;
    auto atoms = mole.read_geometry(filename);
    auto shells = mole.make_sto3g_basis(atoms);

    std::cout << "down build!\n";
    auto start = std::chrono::high_resolution_clock::now();
    mole.m_S = mole.calc_ovlp(shells);
    mole.m_H = mole.calc_core_h(shells, atoms);
    mole.m_I = mole.calc_eri(shells);
    auto end = std::chrono::high_resolution_clock::now();
    
    
    //tensor_print(mole.m_I);
    std::cout << mole.m_I.sum() << std::endl;
    //std::cout << "S:\n" << mole.m_S << "\n";
    /*
    std::cout << "T:\n" << T << "\n";
    std::cout << "V:\n" << V << "\n";
    */

    // 计算并输出耗时
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time to calculate S: " << diff.count() << " s\n";

    return 0;
}

Matrix Mole::calc_ovlp(std::vector<Shell>& shells) {

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

Matrix Mole::calc_kin(std::vector<Shell>& shells) {

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


Matrix Mole::calc_nuc(std::vector<Shell>& shells, std::vector<Atom>& atoms) {

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

Tensor Mole::calc_eri(std::vector<Shell>& shells) {

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