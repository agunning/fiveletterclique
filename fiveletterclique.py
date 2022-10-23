import numpy as np
import time
def run_flc(dict_path ='words_alpha.txt',output_path='fiveletterclique.txt'):
    start=time.perf_counter()
    with open(dict_path,'r') as f:
        words=np.array([X[:-1] for X in f if len(X)==6])
    pi=np.array([ 0, 18, 11, 12,  1, 20, 16, 13,  3, 23, 17,  8, 14,  7,  4, 15, 25,
        5,  2,  9,  6, 21, 19, 24, 10, 22])
    raw=pi[words.view('int32').reshape((-1,5))-97]
    dup  = ((raw.reshape((-1,1,5))==raw.reshape((-1,5,1))).sum(axis=(1,2))<6)
    rmduplicates=raw[dup,:]
    balist=(2**rmduplicates).sum(axis=1)
    balist2,counts=np.unique(balist,return_counts=True)
    words2=words[dup][np.argsort(balist)]
    anagramsl=np.concatenate([[0],np.cumsum(counts)])
    def K(A,B):
        return np.logical_not(B&A.reshape((-1,1)))
    exp=2**np.array(range(0,27))
    dexp=2**26-exp
    rs=np.searchsorted(balist2,exp)
    def addtaboo(words):
        words =words[(2**26*words[:,1]+words[:,0]).argsort()]
        index =words[:,1].sum()
        n_words=np.concatenate([words[-index:],words])
        n_words[:index,1]=0
        qs=np.searchsorted(n_words[:index,0],dexp)
        for i in range(25,-1,-1):
            n_words[qs[i+1]:qs[i],0]+=2**i
            if qs[i]==index:
                break
        return n_words
    def addword(words):
        words=words[words[:,0].argsort()]
        qs=np.searchsorted(words[:,0],dexp)
        l=words.shape[0]
        words_list=[]
        for i in range(25,-1,-1):
            uniq,index,counts=np.unique(words[qs[i+1]:qs[i],0],return_index=True,return_counts=True)
            nz=np.nonzero(K(uniq,balist2[rs[i]:rs[i+1]]))
            if len(nz[0])==0:
                continue
            repeats=counts[nz[0]]
            balistinds=rs[i]+np.repeat(nz[1],repeats)
            currinds=qs[i+1]+np.repeat(index[nz[0]]+np.cumsum(repeats),repeats)-np.arange(1,sum(repeats)+1)
            newarr=np.hstack((words[currinds],balist2[balistinds].reshape((-1,1))))
            newarr[:,0]+=newarr[:,-1]
            words_list.append(newarr.copy())
            if qs[i]==l:
                break
        n_words=np.concatenate(words_list)
        return n_words
    G=np.array([[0,1]])
    for i in range(5):
        G=addword(addtaboo(G))
    outputs=G[:,2:]
    outputs2=np.searchsorted(balist2,outputs)
    V=anagramsl[1:]-anagramsl[:-1]
    wordsquarelist = np.zeros((V[outputs2].prod(axis=1).sum(),5),dtype='<U5')
    t=0
    for i in range(len(outputs)):
        X=np.array(list(np.ndindex(tuple(V[outputs2[i]]))))
        index=anagramsl[outputs2[i]]+X
        wordsquarelist[t:t+X.shape[0]]=words2[index]
        t+=X.shape[0]
    lines = [' '.join(X) for X in wordsquarelist]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(time.perf_counter()-start)
