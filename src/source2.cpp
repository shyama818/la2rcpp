#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Generate Random Fourier Features
// "Random features for large-scale kernel machines, 2008"
// 
// [[Rcpp::export()]]
arma::mat feature(arma::mat& tdata, arma::mat& ran0, const int brocksize){

arma::mat tmp0 = ran0*tdata;

arma::mat phix = arma::join_cols(cos(tmp0),sin(tmp0)) / sqrt(brocksize);

return phix;

}

// Generate Random Fourier Features for MKL
// "Random features for large-scale kernel machines, 2008"
// 
// [[Rcpp::export()]]
arma::mat featurem(arma::mat& tdata, arma::rowvec theta, arma::mat& ran0, const int brocksize, const int odimno, arma::rowvec& start){

int colno = tdata.n_cols;

arma::mat tmp0;
arma::mat phix = arma::zeros(2*brocksize,colno);
for (int ri=0; ri<odimno; ri++){
    int s0 = start(ri)-1;
    int s1 = start(ri+1)-2;
    tmp0 = ran0.cols(s0,s1)*tdata.rows(s0,s1);
    phix += sqrt(theta(ri))*arma::join_cols(cos(tmp0),sin(tmp0)) / sqrt(brocksize);
}
tmp0 = ran0*tdata;
phix += sqrt(theta(odimno))*arma::join_cols(cos(tmp0),sin(tmp0)) / sqrt(brocksize);

return phix;

}

// Generate Random Fourier Features for the i-th original variable
// "Random features for large-scale kernel machines, 2008"
// 
// [[Rcpp::export()]]
arma::mat feature2(arma::mat& tdata, arma::mat& ran0, const int brocksize, arma::rowvec& start, const int ri){

int s0 = start(ri)-1;
int s1 = start(ri+1)-2;
arma::mat tmp0 = ran0.cols(s0,s1)*tdata.rows(s0,s1);
arma::mat phix = arma::join_cols(cos(tmp0),sin(tmp0)) / sqrt(brocksize);

return phix;

}

// PCA (Principal Component Analysis)
// To select eigenvectors corresponding to the largest N eigenvalues
// 
// [[Rcpp::export()]]
arma::mat pca(const int mlimit, arma::mat& traindata0){

    int nc = traindata0.n_cols;
    int subsample_size = std::min(nc, 10000);
    arma::uvec pI = arma::randperm(nc);
    arma::mat stdata0 = traindata0.cols(pI);

    arma::mat stdata = stdata0.cols(0, subsample_size-1);
    arma::mat covmat = stdata * trans(stdata) / subsample_size;
    arma::vec eigval;
    arma::mat eigvec0;
    arma::eig_sym(eigval, eigvec0, covmat);
    arma::mat eigvec1 = fliplr(eigvec0);
    int nr = traindata0.n_rows;
    int output_size = std::min(nr, mlimit);
    arma::mat eigvec = eigvec1.cols(0,output_size-1);

return eigvec;

}

// Median trick
// "Large sample analysis of the median heuristic, 2018"
// 
// [[Rcpp::export()]]
double mtrick(arma::mat& traindata){

int ntr = traindata.n_cols;
int pmin = std::min(ntr,2000);
arma::uvec pI = arma::randperm(ntr);
arma::mat ptdata0 = traindata.cols(pI);

arma::mat ptdata = ptdata0.cols(0, pmin-1);

int iterm = (pmin-1)*pmin/2;
arma::mat med0 = arma::zeros(1,iterm);
int k=-1;
for (int i=0; i<(pmin-1); i++) {
    for (int j=i+1; j<pmin; j++) {
        k=k+1;
        arma::vec dif = ptdata.col(i) - ptdata.col(j);
        med0(0,k) = arma::norm(dif);
    }
}
double s_coeff = 1;
arma::vec med1 = arma::median(med0,1);
double s = 1 / std::pow((s_coeff * med1(0)),2);


return s;

}

// Compute softmax
//
// [[Rcpp::export()]]
arma::mat softmax(arma::mat& train_batch_preds){

    int k_n = train_batch_preds.n_rows;
    arma::rowvec max_y0 = arma::max(train_batch_preds,0);
    arma::mat max_y = arma::repmat(max_y0,k_n,1);
    arma::mat ny = arma::exp(train_batch_preds - max_y);
    arma::rowvec sum_ny0 = arma::sum(ny,0);
    arma::mat sum_ny = arma::repmat(sum_ny0,k_n,1);
    arma::mat softmax = ny / sum_ny;

return softmax;

}

// Update W
//
// [[Rcpp::export()]]
arma::mat update_w(const double step_size, const double reg_param, const int batch_size, const int blocksz, arma::mat& residue, arma::mat& W_mat, arma::mat& train_batch_X){

    arma::mat covx = train_batch_X * arma::trans(train_batch_X) / batch_size;
    arma::mat preconditioner = covx + (reg_param + 1e-7) * arma::eye(2*blocksz, 2*blocksz);

    arma::mat updateW = - step_size * (residue * arma::trans(train_batch_X) / batch_size + reg_param * W_mat ) * arma::inv_sympd(preconditioner);

return updateW;
}

// Update W
//
// [[Rcpp::export()]]
arma::rowvec mapvec(arma::rowvec x, arma::rowvec z1, arma::rowvec z2){

    arma::rowvec y(x.n_elem);

    for (int i=0;i<x.n_elem;i++){
        for (int j=0;j<z1.n_elem;j++){
            int p1 = std::round(x(i)*std::pow(10,8));
            int p2 = std::round(z1(j)*std::pow(10,8));
            if ( p1 == p2 ){
                y(i) = z2(j);
            }
        }
    }

    return y;

}

// find a row whose entries are identical to thos of the vector
// return the row number if it is found and -1 if there are no matching rows.
//
// [[Rcpp::export()]]
int row_find(arma::mat x, arma::rowvec y){

    int n0 = x.n_rows;
    int findex = -1;

    for (int k=0;k<n0;k++){
         if ( arma::approx_equal(x.row(k), y, "absdiff", 1e-8) ){
               findex = k;
               break;
         }
    }

    return findex;

}

// delete redundant rows.
//
// [[Rcpp::export()]]
arma::mat uniqueVec(arma::mat& x){

    int n0 = x.n_rows;

    arma::mat uniquemat = x.row(0);

    if (n0 > 1){
        int frow;
        for (int k=1;k<n0;k++){
            frow = row_find(x.rows(0,k-1), x.row(k));
            if (frow == -1){
                uniquemat = arma::join_vert( uniquemat, x.row(k));
            }
        }
    }

    return uniquemat;

}

