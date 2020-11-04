# vim: et sts=4 ts=4
import argparse
import numpy as np
from supersvd import supersvd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['real', 'double'], default='real',
                        help="Data type, default is '%(default)s'")
    parser.add_argument("-x", metavar="X.STD", required=True, help="X data input file name")
    parser.add_argument("-y", metavar="Y.STD", required=True, help="Y data input file name")
    parser.add_argument("-t", "--time", type=int, required=True, help="Length of the time interval")
    parser.add_argument("-k", type=int, default=3,
                        help="Number of singular values, default is %(default)d")
    parser.add_argument("-xv", help="X singular vectors output file name, if necessary")
    parser.add_argument("-yv", help="Y singular vectors output file name, if necessary")
    parser.add_argument("-xc", help="X time coefficients output file name, if necessary")
    parser.add_argument("-yc", help="Y time coefficients output file name, if necessary")
    parser.add_argument("-stat", help="Correlation and variance values in CSV, if necessary")
    parser.add_argument("--dont-subtract-mean", dest="elim_mean", help="Disable subtracting of the time mean from input", action="store_false")
    args = parser.parse_args()

    if args.type == 'real':
        dtype = np.float32
    else:
        dtype = np.float64
   
    t = args.time

    X = np.fromfile(args.x, dtype=dtype).reshape(t, -1)
    Y = np.fromfile(args.y, dtype=dtype).reshape(t, -1)

    svd = supersvd(X, Y, args.k, args.elim_mean)
    
    if args.xv is not None:
        svd.x_vect.tofile(args.xv)
    if args.yv is not None:
        svd.y_vect.tofile(args.yv)
    if args.xc is not None:
        svd.x_coeff.tofile(args.xc)
    if args.yc is not None:
        svd.y_coeff.tofile(args.yc)
    
    if args.stat is not None:
        f = open(args.stat, 'w')
        f.write('number,corrcoeff,x_varfrac,y_varfrac,covfrac\n')
    for i in range(args.k):
        print('Singular value number:', i+1)
        corrcoeff = 100 * svd.corrcoeff[i]
        x_varfrac = 100 * svd.x_variance_fraction[i]
        y_varfrac = 100 * svd.y_variance_fraction[i]
        covfrac = 100 * svd.eigenvalue_fraction[i]

        if args.stat is not None:
            f.write('%d,%f,%f,%f,%f\n' % (i+1, corrcoeff, x_varfrac, y_varfrac, covfrac))
        print('Time series correlation coefficient:', '%8.4f%%' % corrcoeff)
        print(args.x, 'variance fraction:', '%8.4f%%' % x_varfrac)
        print(args.y, 'variance fraction:', '%8.4f%%' % y_varfrac)
        print('Covariance fraction:', '%8.4f%%' % covfrac)

    if args.stat is not None:
        f.close()

    

if __name__ == "__main__":
    main()
