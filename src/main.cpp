#include "Mole.h"

int main(int argc, char* argv[]) {
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    Mole mole;
    auto atoms = mole.read_geometry(filename);
    auto shells = mole.make_sto3g_basis(atoms);

    auto start = std::chrono::high_resolution_clock::now();
    mole.m_S = mole.calc_ovlp(shells);
    mole.m_H = mole.calc_core_h(shells, atoms);
    mole.m_I = mole.calc_eri(shells);
    //std::cout << I.sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    
    /*
    std::cout << "S:\n" << S << "\n";
    std::cout << "T:\n" << T << "\n";
    std::cout << "V:\n" << V << "\n";
    */

    // 计算并输出耗时
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time to calculate S: " << diff.count() << " s\n";

    return 0;
}



Matrix Mole::calc_ovlp(std::vector<Shell>& shells) {

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Matrix S = Matrix::Zero(nao, nao);
    //m_S.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normB, shells[jpr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA ) {
                        auto ib = 0;
                        for (auto cb : coeffB) {

                            buf(fx, fy) += normA[ia] * normB[ib] * ca * cb * \
                                        overlap_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord);
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    S(x + i, y + j) = buf(i, j);
                    S(y + j, x + i) = buf(i, j);
                }
            }
            //std::cout << "buf" << buf << std::endl<< std::endl;
            //std::cout <<"S" << S << std::endl << std::endl;
        }  
    }
    
    return S;
}

Matrix Mole::calc_kin(std::vector<Shell>& shells) {

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }
    Matrix T = Matrix::Zero(nao, nao);
    //m_T.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normB, shells[jpr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA) {
                        auto ib = 0;
                        for (auto cb : coeffB) {

                            buf(fx, fy) += normA[ia] * normB[ib] * ca * cb * \
                                        kinetic_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord);
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    T(x + i, y + j) = buf(i, j);
                    T(y + j, x + i) = buf(i, j);
                }
            }
        }  
    }
    
    return T;
}


Matrix Mole::calc_nuc(std::vector<Shell>& shells, std::vector<Atom>& atoms) {

    std::vector<std::pair<int, std::vector<double>>> q;
    for (const auto& atom : atoms) {
         q.push_back({static_cast<int>(atom.atomic_number),
                   {{atom.x, atom.y, atom.z}}});
    }

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Matrix V = Matrix::Zero(nao, nao);
    //m_V.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normA, shells[ipr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA) {
                        auto ib = 0;
                        for (auto cb : coeffB) { 
                            for (auto nuc_cent : q) {   

                                buf(fx, fy) += -nuc_cent.first * normA[ia] * normB[ib] * ca * cb * \
                                        nuclear_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord, nuc_cent.second);
                            
                            }  
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    V(x + i, y + j) = buf(i, j);
                    V(y + j, x + i) = buf(i, j);
                }
            }
        }  
    }
    
    return V;
}


Tensor Mole::calc_eri(std::vector<Shell>& shells) {
    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Tensor I(nao, nao, nao ,nao);
    I.setZero();
    for (int ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (int jpr = 0; jpr <= ipr; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (int i = 0; i < jpr; i++) {
            y += shells[i].ngto;
            }
            for (int kpr = 0; kpr <= ipr; kpr++) {
                auto dk = shells[kpr].ngto;
                auto z = 0;
                for (int i = 0; i < kpr; i++) {
                    z += shells[i].ngto;
                }
                const auto lpr_max = (ipr == kpr) ? jpr : kpr;
                for (int lpr = 0; lpr <=  lpr_max; lpr++) {
                    auto dl = shells[lpr].ngto;
                    auto w = 0;
                    for (int i = 0; i < lpr; i++) {
                        w += shells[i].ngto;
                    }
                    Tensor buf(di, dj, dk, dl);
                    buf.setZero();
                    //std::cout << "(" << ipr << jpr <<"|" << kpr << lpr << ")" << std::endl;

                    auto fx = 0;
                    for (auto lmn1 : shells[ipr].shl) {
                        auto normA = norm(lmn1, shells[ipr].exp);
                        auto coeffA = shells[ipr].coeff;
                        normalization(coeffA, lmn1, normA, shells[ipr].exp);
                        auto fy = 0;
                        for (auto lmn2 : shells[jpr].shl) {
                            auto normB = norm(lmn2, shells[jpr].exp);
                            auto coeffB = shells[jpr].coeff;
                            normalization(coeffB, lmn2, normB, shells[jpr].exp);
                            auto fz = 0;
                            for (auto lmn3 : shells[kpr].shl) {
                                auto normC = norm(lmn3, shells[kpr].exp);
                                auto coeffC = shells[kpr].coeff;
                                normalization(coeffC, lmn3, normC, shells[kpr].exp);
                                auto fw = 0;
                                for (auto lmn4 : shells[lpr].shl) {
                                    auto normD = norm(lmn4, shells[lpr].exp);
                                    auto coeffD = shells[lpr].coeff;
                                    normalization(coeffD, lmn4, normD, shells[lpr].exp);
                                    
                                    auto ia = 0;
                                    for (auto ca : coeffA) {
                                        auto ib = 0;
                                        for (auto cb : coeffB) { 
                                            auto ic = 0;
                                            for (auto cc : coeffC) {
                                                auto id = 0;
                                                for (auto cd : coeffD) {
                                                    buf(fx, fy, fz, fw) += normA[ia] * normB[ib] * normC[ic] * normD[id] * \
                                                                            ca * cb * cc * cd * \
                                                                            electron_repulsion(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                                                                               shells[jpr].exp[ib], lmn2, shells[jpr].coord,
                                                                                               shells[kpr].exp[ic], lmn3, shells[kpr].coord,
                                                                                               shells[lpr].exp[id], lmn4, shells[lpr].coord);
                                                id++;    
                                                }
                                            ic++;    
                                            }
                                        ib++;
                                        }
                                    ia++;
                                    }
                                fw++;
                                }
                            fz++;
                            }
                        fy++;
                        }
                    fx++;
                    }  
                           
                    for (int i = 0; i < di; ++i) {
                        for (int j = 0; j < dj; ++j) {
                            for (int k = 0; k < dk; ++k) {
                                for (int l = 0; l < dl; l++) {
                                    I(x+i, y+j, z+k, w+l) = buf(i, j, k, l);
                                    I(y+j, x+i, z+k, w+l) = buf(i, j, k, l);
                                    I(x+i, y+j, w+l, z+k) = buf(i, j, k, l);
                                    I(y+j, x+i, w+l, z+k) = buf(i, j, k, l);
                                    I(z+k, w+l, x+i, y+j) = buf(i, j, k, l);
                                    I(w+l, z+k, x+i, y+j) = buf(i, j, k, l);
                                    I(z+k, w+l, y+j, x+i) = buf(i, j, k, l);
                                    I(w+l, z+k, y+j, x+i) = buf(i, j, k, l);
                                }
                            }
                        }  
                    }
                }
            } 
        }
    }
    
    return I;
}