import numpy as np



def tenfold_binary(gene_e, bit=9, t=10):
        out=[]
        for i in gene_e:
            i=float("%.1f" % i)
            a=[0]*bit
            if i>0:
                a[0]=1
                a[-len(bin(int(i*t))[2:]):] = [int(j) for j in bin(int(i*t))[2:]]
                
            if i<0:
                
                a[-len(bin(int(i*t))[3:]):] = [int(j) for j in bin(int(i*t))[3:]]
            out.append(a)
        return np.array(out)

    
def binary(gene_e,bit=10, t=10):
        out=[]
        for i in gene_e:
            i=float("%.1f" % i)
            a=[0]*bit
            if i>0:
                a[0]=1
                a[6-len(bin(int(i))[2:]):6] = [int(j) for j in bin(int(i))[2:]]
                if i-int(i)!=0:
                    ii=int(i*t-int(i)*t)
                    a[-len(bin(int(ii))[2:]):] = [int(j) for j in bin(int(ii))[2:]]
            if i<0:
                
                a[6-len(bin(int(i))[3:]):6] = [int(j) for j in bin(int(i))[3:]]
                if i-int(i)!=0:
                    ii=int(i*t-int(i)*t)
                    a[-len(bin(int(ii))[3:]):] = [int(j) for j in bin(int(ii))[3:]]
            out.append(a)
        return np.array(out)
    
def one_hot(gene_e,bit=25, t=10):
        out=[]
        for i in gene_e:
            i=float("%.1f" % i)
            a=[0]*bit
            if i>0:
                a[0]=1
                a[int(i)] = 1
                if i-int(i)!=0:
                    a[int(i*t-int(i)*t)+15] = 1
            if i<0:
                
                a[abs(int(i))] = 1
                if i-int(i)!=0:
                    a[abs(int(i*t-int(i)*t))+15] = 1
            out.append(a)
        return np.array(out)
    
def value(gene_e):
        out=[]
        for i in g:
            i=float("%.1f" % i)
            a=[i]
            out.append(a)
        return np.array(out)