import numpy as np
import pandas as pd
from .sascorer import calculateScore
from tqdm import tqdm
import selfies as sf
from rdkit import rdBase, Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem.Scaffolds import MurckoScaffold
import seaborn as sns
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors
from rdkit import DataStructs


def valid(l): 
    val = 0
    unval_index = []
    for i in range(len(l)):
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(l[i]), isomericSmiles=True)
            val += 1
        except:
            print("not successfully processed smiles: ", l[i])
            unval_index.append(i)
            pass
    valid = val/len(l)
    return valid, unval_index

def unique(l):
    l_set = list(set(l))
    uniq = len(l_set)/len(l)
    return l_set, uniq

def IntDivp(out):
    all_div = []
    for i in tqdm(range(len(out)-1)):
        div = 0.0
        tot = 0
        for j in range(i+1, len(out)):
            ms = [Chem.MolFromSmiles(out[i]), Chem.MolFromSmiles(out[j])]
            mfp = [Chem.RDKFingerprint(x) for x in ms]
            div += 1 - DataStructs.FingerprintSimilarity(mfp[0], mfp[1],                                                      metric=eval(metic_list[0]))
            tot +=1
        div /= tot
        all_div.append(div)
    all_div = np.array(all_div)
    return all_div

def Qed(out_set):
    qed_list = []
    qed_scores = []
    for i in out_set:
        mol = Chem.MolFromSmiles(i)
        qed_score = QED.default(mol)
        qed = QED.properties(mol)
        qed_scores.append(qed_score)
        qed_list.append(qed)
    return qed_list, qed_scores

def SA(out_set):
    SA_scores = []
    for i in out_set:
        mol = Chem.MolFromSmiles(i)
        SA_score = calculateScore(mol)
        SA_scores.append(SA_score)
    return SA_scores

def Fragment_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
    fraggle_similarity = []
    smi_frag = []
    for i in tqdm(range(len(list_t))):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_t[i])
        try:
            (smi, match) = FraggleSim.GetFraggleSimilarity(mol1,mol2)
            fraggle_similarity.append(smi)
            smi_frag.append(match)
        except:
            fraggle_similarity.append(0)
            smi_frag.append("None")
    return fraggle_similarity, smi_frag

def Fragment_similarity_(list_p, list_n):
    print("输入顺序：predict_list, true_list")
    fraggle_similarity = []
    sim_frag = []
    for i in tqdm(range(len(list_p))):
        a,b=[],[]
        mol1 = Chem.MolFromSmiles(list_p[i])
        for j in range(len(list_n)):
            mol2 = Chem.MolFromSmiles(list_n[j])
            try:
                (smi, match) = FraggleSim.GetFraggleSimilarity(mol1,mol2)
                a.append(smi)
                b.append(match)
            except:
                a.append(0)
                b.append("None")
        fraggle_similarity.append(a)
        sim_frag.append(b)
    return fraggle_similarity, sim_frag

def Scaffold_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
    scaff_sim = []
    for i in range(len(list_p)):
        mol1_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_p[i]))
        mol2_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_t[i]))
        mfp = [Chem.RDKFingerprint(x) for x in [mol1_scaff, mol2_scaff]]
        smi = DataStructs.FingerprintSimilarity(mfp[0], mfp[1], metric=eval(metic_list[2]))
        scaff_sim.append(smi)
    return scaff_sim

def Scaffold_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    scaff_sim = []
    for i in tqdm(range(len(list_p))):
        a=[]
        mol1_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_p[i]))
        for j in range(len(list_n)):
            mol2_scaff = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(list_n[j]))                
            mfp = [Chem.RDKFingerprint(x) for x in [mol1_scaff, mol2_scaff]]
            smi = DataStructs.FingerprintSimilarity(mfp[0], mfp[1], metric=eval(metic_list[2]))
            a.append(smi)
        scaff_sim.append(a)
    return scaff_sim

def ECFP_Tanimoto_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    out=[]
    for i in tqdm(range(len(list_p))):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_n[j])
        ECFPs = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
        out.append(DataStructs.FingerprintSimilarity(ECFPs[0], ECFPs[1], metric=eval(metic_list[0])))
    return out

def ECFP_Tanimoto_similarity_(list_p, list_n):
    print("输入顺序：predict_list, nsclc_list")
    out=[]
    for i in tqdm(range(len(list_p))):
        a=[]
        mol1 = Chem.MolFromSmiles(list_p[i])
        for j in range(len(list_n)):
            mol2 = Chem.MolFromSmiles(list_n[j])
            ECFPs = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
            a.append(DataStructs.FingerprintSimilarity(ECFPs[0], ECFPs[1], metric=eval(metic_list[0])))
        out.append(a)
    return out

def Fingerprint_similarity(list_p, list_t):
    print("输入顺序：predict_list, true_list")
#     RDKfp_sim,MACCS_sim,AP_sim,tts_sim,MGfp_sim,ECFP4_sim,FCFP4_sim = [],[],[],[],[],[],[]
    RDKfp_sim, MACCS_sim, MGfp_sim = [], [], []
    for i in range(len(list_p)):
        mol1 = Chem.MolFromSmiles(list_p[i])
        mol2 = Chem.MolFromSmiles(list_t[i])
        RDK_fps = [Chem.RDKFingerprint(x) for x in [mol1, mol2]]
        MACCS_fps = [MACCSkeys.GenMACCSKeys(x) for x in [mol1, mol2]]
#         AP_fps = [Pairs.GetAtomPairFingerprint(x) for x in [mol1, mol2]]
#         tts_fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(x) for x in [mol1, mol2]]
        MG_fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,2048) for x in [mol1, mol2]]
#         ECFP4_fps = [Chem.AllChem.GetMorganFingerprint(x,2) for x in [mol1, mol2]]
#         FCFP4_fps = [Chem.AllChem.GetMorganFingerprint(x,2, useFeatures=True) for x in [mol1, mol2]]
        RDKfp_sim.append(DataStructs.FingerprintSimilarity(RDK_fps[0], RDK_fps[1], metric=eval(metic_list[0])))
        MACCS_sim.append(DataStructs.FingerprintSimilarity(MACCS_fps[0], MACCS_fps[1], metric=eval(metic_list[1])))
#         AP_sim.append(DataStructs.FingerprintSimilarity(AP_fps[0], AP_fps[1], metric=eval(metic_list[0]))) 
#         tts_sim.append(DataStructs.FingerprintSimilarity(tts_fps[0], tts_fps[1], metric=eval(metic_list[0])))
        MGfp_sim.append(DataStructs.FingerprintSimilarity(MG_fps[0], MG_fps[1], metric=eval(metic_list[1])))
#         ECFP4_sim.append(DataStructs.FingerprintSimilarity(ECFP4_fps[0], ECFP4_fps[1], metric=eval(metic_list[1])))
#         FCFP4_sim.append(DataStructs.FingerprintSimilarity(FCFP4_fps[0], FCFP4_fps[1], metric=eval(metic_list[1])))
    return RDKfp_sim, MACCS_sim, MGfp_sim

def squeeze(lt):
    out=[]
    for i in lt:
        for j in i:
            out.append(j)
    return out

def to_index(i, n):
    return(i//n, i%n)

metic_list = ['DataStructs.TanimotoSimilarity', 'DataStructs.DiceSimilarity',
            'DataStructs.CosineSimilarity', 'DataStructs.SokalSimilarity',
            'DataStructs.RusselSimilarity', 'DataStructs.KulczynskiSimilarity',
             'DataStructs.McConnaugheySimilarity']