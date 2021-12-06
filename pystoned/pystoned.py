import os
import selfies
import rdkit
import random
import numpy as np
import random
from random import randrange
from rdkit import Chem
from selfies import encoder, decoder
from rdkit.Chem import CanonSmiles, MolFromSmiles as smi2mol
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import Mol
from rdkit.Chem import Descriptors
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import Draw
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from fingerprints import get_fingerprint, get_fp_scores


class ChemicalSubspace():

    def __init__(self, smiles, num_random_samples=100, num_mutation_ls=[1, 2, 3, 4, 5], 
                fp_type="ECFP4", preserve_substructure_smiles=None):
        self.smiles = smiles
        self.num_random_samples = num_random_samples
        self.num_mutation_ls = num_mutation_ls
        self.fp_type = fp_type
        self.preserve_substructure = False if preserve_substructure_smiles is None else True
        self.preserve_substructure_smiles = preserve_substructure_smiles

        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol == None: 
            raise Exception('Invalid starting structure encountered')

        if preserve_substructure_smiles is not None:
            self.mol_substructure = Chem.MolFromSmiles(preserve_substructure_smiles)
            if self.mol_substructure == None: 
                raise Exception('Invalid substructure encountered')

    def generate(self):
        # step1 : randomize smiles
        self.randomize_smile_orderings = self.get_reorder_smiles()

        # step2 : convert smiles to selfies
        self.selfies_ls = self.convert_smiles2selfies(self.randomize_smile_orderings)

        # step3-4 : mutate selfies & convert back to smiles
        self.all_smiles_collect, _ = self.perform_random_mutations(self.selfies_ls)

        # sanitize smiles
        self.canon_smi_ls = self.get_sanitized_smiles(self.all_smiles_collect)

        return self.canon_smi_ls

    def get_reorder_smiles(self):
        return [self.randomize_smiles(self.mol) for _ in range(self.num_random_samples)]

    def convert_smiles2selfies(self, randomize_smile_orderings):
        return [encoder(x) for x in randomize_smile_orderings]

    def perform_random_mutations(self, selfies_ls):
        all_smiles_collect = []
        all_smiles_collect_broken = []

        for num_mutations in self.num_mutation_ls: 
            # Mutate the SELFIES: 
            selfies_mut = self.get_mutated_SELFIES(selfies_ls.copy(), num_mutations=num_mutations)

            # Convert back to SMILES: 
            smiles_back = [decoder(x) for x in selfies_mut]
            all_smiles_collect = all_smiles_collect + smiles_back
            all_smiles_collect_broken.append(smiles_back)

        return all_smiles_collect, all_smiles_collect_broken

    def get_sanitized_smiles(self, all_smiles_collect):
        canon_smi_ls = []
        for item in all_smiles_collect: 
            mol, smi_canon, did_convert = self.sanitize_smiles(item)
            if mol == None or smi_canon == '' or did_convert == False: 
                raise Exception('Invalid smile string found')
            canon_smi_ls.append(smi_canon)
        canon_smi_ls = list(set(canon_smi_ls))
        return canon_smi_ls

    def substructure_preserver(self, mol):
        """
        Check for substructure violates
        Return True: contains a substructure violation
        Return False: No substructure violation
        """       
        if mol.HasSubstructMatch(rdkit.Chem.MolFromSmarts(self.preserve_substructure_smiles)) == True:
            return True # The has substructure! 
        else: 
            return False # Molecule does not have substructure!

    def randomize_smiles(self, mol):
        '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.
        Parameters:
        mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)

        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
        '''
        if not mol:
            return None

        Chem.Kekulize(mol)
        return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True) 

    def sanitize_smiles(self, smi):
        '''Return a canonical smile representation of smi

        Parameters:
        smi (string) : smile string to be canonicalized 

        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
        smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
        conversion_successful (bool): True/False to indicate if conversion was  successful 
        '''
        try:
            mol = smi2mol(smi, sanitize=True)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            return (mol, smi_canon, True)
        except:
            return (None, None, False)

    def get_selfie_chars(self, selfie):
        '''Obtain a list of all selfie characters in string selfie

        Parameters: 
        selfie (string) : A selfie string - representing a molecule 

        Example: 
        >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
        ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

        Returns:
        chars_selfie: list of selfie characters present in molecule selfie
        '''
        chars_selfie = [] # A list of all SELFIE sybols from string selfie
        while selfie != '':
            chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
            selfie = selfie[selfie.find(']')+1:]
        return chars_selfie

    def mutate_selfie(self, selfie, max_molecules_len, write_fail_cases=False):
        '''Return a mutated selfie string (only one mutation on slefie is performed)

        Mutations are done until a valid molecule is obtained 
        Rules of mutation: With a 33.3% propbabily, either: 
            1. Add a random SELFIE character in the string
            2. Replace a random SELFIE character with another
            3. Delete a random character

        Parameters:
        selfie            (string)  : SELFIE string to be mutated 
        max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
        write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"

        Returns:
        selfie_mutated    (string)  : Mutated SELFIE string
        smiles_canon      (string)  : canonical smile of mutated SELFIE string
        '''
        valid=False
        fail_counter = 0
        chars_selfie = self.get_selfie_chars(selfie)

        while not valid:
            fail_counter += 1

            alphabet = list(selfies.get_semantic_robust_alphabet()) # 34 SELFIE characters 

            choice_ls = [1, 2, 3] # 1=Insert; 2=Replace; 3=Delete
            random_choice = np.random.choice(choice_ls, 1)[0]

            # Insert a character in a Random Location
            if random_choice == 1: 
                random_index = np.random.randint(len(chars_selfie)+1)
                random_character = np.random.choice(alphabet, size=1)[0]

                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

            # Replace a random character 
            elif random_choice == 2:                         
                random_index = np.random.randint(len(chars_selfie))
                random_character = np.random.choice(alphabet, size=1)[0]
                if random_index == 0:
                    selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]

            # Delete a random character
            elif random_choice == 3: 
                random_index = np.random.randint(len(chars_selfie))
                if random_index == 0:
                    selfie_mutated_chars = chars_selfie[random_index+1:]
                else:
                    selfie_mutated_chars = chars_selfie[:random_index] + chars_selfie[random_index+1:]

            else: 
                raise Exception('Invalid Operation trying to be performed')

            selfie_mutated = "".join(x for x in selfie_mutated_chars)
            sf = "".join(x for x in chars_selfie)

            try:
                smiles = decoder(selfie_mutated)
                mol, smiles_canon, done = self.sanitize_smiles(smiles)
                if self.preserve_substructure:
                    if len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="" or self.substructure_preserver(mol)==False:
                        done = False
                    elif  len(selfie_mutated_chars) > max_molecules_len or smiles_canon=="":
                        done = False

                if done:
                    valid = True
                else:
                    valid = False
            except:
                valid=False
                if fail_counter > 1 and write_fail_cases == True:
                    f = open("selfie_failure_cases.txt", "a+")
                    f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                    f.close()

        return (selfie_mutated, smiles_canon)

    def get_mutated_SELFIES(self, selfies_ls, num_mutations): 
        ''' Mutate all the SELFIES in 'selfies_ls' 'num_mutations' number of times. 

        Parameters:
        selfies_ls   (list)  : A list of SELFIES 
        num_mutations (int)  : number of mutations to perform on each SELFIES within 'selfies_ls'

        Returns:
        selfies_ls   (list)  : A list of mutated SELFIES

        '''
        for _ in range(num_mutations): 
            selfie_ls_mut_ls = []
            for str_ in selfies_ls: 

                str_chars = self.get_selfie_chars(str_)
                max_molecules_len = len(str_chars) + num_mutations

                selfie_mutated, _ = self.mutate_selfie(str_, max_molecules_len)
                selfie_ls_mut_ls.append(selfie_mutated)

            selfies_ls = selfie_ls_mut_ls.copy()
        return selfies_ls




class ChemicalPath():

    def __init__(self, starting_smiles, target_smiles, num_tries=2, num_random_samples=2, 
                collect_bidirectional=True, fp_type="ECFP4"):
        self.starting_smiles = starting_smiles
        self.target_smiles = target_smiles
        self.num_tries = num_tries
        self.num_random_samples = num_random_samples
        self.collect_bidirectional = collect_bidirectional
        self.fp_type = fp_type

        self.mol = Chem.MolFromSmiles(starting_smiles)
        if self.mol == None: 
            raise Exception('Invalid starting structure encountered')

        self.mol = Chem.MolFromSmiles(target_smiles)
        if self.mol == None: 
            raise Exception('Invalid target structure encountered')

    def get_median_mols(self, num_top_iter=12):
        smiles_paths_dir1, smiles_paths_dir2 = self.generate()
    
        # Find the median molecule & plot: 
        all_smiles_dir_1 = [item for sublist in smiles_paths_dir1 for item in sublist] # all the smile string of dir1
        all_smiles_dir_2 = [item for sublist in smiles_paths_dir2 for item in sublist] # all the smile string of dir2
    
        all_smiles = [] # Collection of valid smile strings 
        for smi in all_smiles_dir_1 + all_smiles_dir_2: 
            if Chem.MolFromSmiles(smi) != None: 
                mol, smi_canon, _ = self.sanitize_smiles(smi)
                all_smiles.append(smi_canon)

        all_smiles = list(set(all_smiles))

        scores_start  = get_fp_scores(all_smiles, self.starting_smiles, self.fp_type)   # similarity to target
        scores_target = get_fp_scores(all_smiles, self.target_smiles, self.fp_type)     # similarity to starting structure
        data          = np.array([scores_target, scores_start])
        avg_score     = np.average(data, axis=0)
        better_score  = avg_score - (np.abs(data[0] - data[1]))   
        better_score  = ((1/9) * better_score**3) - ((7/9) * better_score**2) + ((19/12) * better_score)
    
        best_idx = better_score.argsort()[-num_top_iter:][::-1]
        best_smi = [all_smiles[i] for i in best_idx]
        best_scores = [better_score[i] for i in best_idx]

        return best_smi, best_scores  

    def generate(self):
        return self.get_compr_paths(starting_smile=self.starting_smiles,
                                    target_smile=self.target_smiles,
                                    num_tries=self.num_tries,
                                    num_random_samples=self.num_random_samples,
                                    collect_bidirectional=self.collect_bidirectional)

    def get_compr_paths(self, starting_smile, target_smile, num_tries, num_random_samples, collect_bidirectional):
        ''' Obtaining multiple paths/chemical paths from starting_smile to target_smile. 

        Parameters:
        starting_smile (string)     : SMILES string (needs to be a valid molecule)
        target_smile (int)          : SMILES string (needs to be a valid molecule)
        num_tries (int)             : Number of path/chemical path attempts between the exact same smiles
        num_random_samples (int)    : Number of different SMILES string orderings to conside for starting_smile & target_smile 
        collect_bidirectional (bool): If true, forms paths from target_smiles-> target_smiles (doubles number of paths)

        Returns:
        smiles_paths_dir1 (list): list paths containing smiles in path between starting_smile -> target_smile
        smiles_paths_dir2 (list): list paths containing smiles in path between target_smile -> starting_smile
        '''
        starting_smile_rand_ord = self.get_random_smiles(starting_smile, num_random_samples=num_random_samples)
        target_smile_rand_ord   = self.get_random_smiles(target_smile,   num_random_samples=num_random_samples)

        smiles_paths_dir1 = [] # All paths from starting_smile -> target_smile
        for smi_start in starting_smile_rand_ord: 
            for smi_target in target_smile_rand_ord: 

                if Chem.MolFromSmiles(smi_start) == None or Chem.MolFromSmiles(smi_target) == None: 
                    raise Exception('Invalid structures')

                for _ in range(num_tries): 
                    path, _, _, _ = self.obtain_path(smi_start, smi_target, filter_path=True)
                    smiles_paths_dir1.append(path)

        smiles_paths_dir2 = [] # All paths from target_smile -> starting_smile
        if collect_bidirectional == True: 
            starting_smile_rand_ord = self.get_random_smiles(target_smile, num_random_samples=num_random_samples)
            target_smile_rand_ord   = self.get_random_smiles(starting_smile,   num_random_samples=num_random_samples)

            for smi_start in starting_smile_rand_ord: 
                for smi_target in target_smile_rand_ord: 

                    if Chem.MolFromSmiles(smi_start) == None or Chem.MolFromSmiles(smi_target) == None: 
                        raise Exception('Invalid structures')

                    for _ in range(num_tries): 
                        path, _, _, _ = self.obtain_path(smi_start, smi_target, filter_path=True)
                        smiles_paths_dir2.append(path)

        return smiles_paths_dir1, smiles_paths_dir2

    def obtain_path(self, starting_smile, target_smile, filter_path=False): 
        ''' Obtain a path/chemical path from starting_smile to target_smile

        Parameters:
        starting_smile (string) : SMILES string (needs to be a valid molecule)
        target_smile (int)      : SMILES string (needs to be a valid molecule)
        filter_path (bool)      : If True, a chemical path is returned, else only a path

        Returns:
        path_smiles (list)                  : A list of smiles in path between starting_smile & target_smile
        path_fp_scores (list of floats)     : Fingerprint similarity to 'target_smile' for each smiles in path_smiles
        smiles_path (list)                  : A list of smiles in CHEMICAL path between starting_smile & target_smile (if filter_path==False, then empty)
        filtered_path_score (list of floats): Fingerprint similarity to 'target_smile' for each smiles in smiles_path (if filter_path==False, then empty)
        '''
        starting_selfie = encoder(starting_smile)
        target_selfie   = encoder(target_smile)

        starting_selfie_chars = self.get_selfie_chars(starting_selfie)
        target_selfie_chars   = self.get_selfie_chars(target_selfie)

        # Pad the smaller string
        if len(starting_selfie_chars) < len(target_selfie_chars): 
            for _ in range(len(target_selfie_chars)-len(starting_selfie_chars)):
                starting_selfie_chars.append(' ')
        else: 
            for _ in range(len(starting_selfie_chars)-len(target_selfie_chars)):
                target_selfie_chars.append(' ')

        indices_diff = [i for i in range(len(starting_selfie_chars)) if starting_selfie_chars[i] != target_selfie_chars[i]]
        path         = {}
        path[0]  = starting_selfie_chars

        for iter_ in range(len(indices_diff)): 
            idx = np.random.choice(indices_diff, 1)[0] # Index to be operated on
            indices_diff.remove(idx)                   # Remove that index

            # Select the last member of path: 
            path_member = path[iter_].copy()

            # Mutate that character to the correct value: 
            path_member[idx] = target_selfie_chars[idx]
            path[iter_+1] = path_member.copy()

        # Collapse path to make them into SELFIE strings
        paths_selfies = []
        for i in range(len(path)):
            selfie_str = ''.join(x for x in path[i])
            paths_selfies.append(selfie_str.replace(' ', ''))

        if paths_selfies[-1] != target_selfie: 
            raise Exception("Unable to discover target structure!")

        # Obtain similarity scores, and only choose the increasing members: 
        path_smiles         = [decoder(x) for x in paths_selfies]
        path_fp_scores      = []
        filtered_path_score = []
        smiles_path         = []

        if filter_path: 
            path_fp_scores = get_fp_scores(path_smiles, target_smile, self.fp_type)

            filtered_path_score = []
            smiles_path   = []
            for i in range(1, len(path_fp_scores)-1): 
                if i == 1: 
                    filtered_path_score.append(path_fp_scores[1])
                    smiles_path.append(path_smiles[i])
                    continue
                if filtered_path_score[-1] < path_fp_scores[i]:
                    filtered_path_score.append(path_fp_scores[i])
                    smiles_path.append(path_smiles[i])

        return path_smiles, path_fp_scores, smiles_path, filtered_path_score

    def get_random_smiles(self, smi, num_random_samples): 
        ''' Obtain 'num_random_samples' non-unique SMILES orderings of smi

        Parameters:
        smi (string)            : Input SMILES string (needs to be a valid molecule)
        num_random_samples (int): Number fo unique different SMILES orderings to form 

        Returns:
        randomized_smile_orderings (list) : list of SMILES strings
        '''
        mol = Chem.MolFromSmiles(smi)
        if mol == None: 
            raise Exception('Invalid starting structure encountered')
        randomized_smile_orderings  = [self.randomize_smiles(mol) for _ in range(num_random_samples)]
        randomized_smile_orderings  = list(set(randomized_smile_orderings)) # Only consider unique SMILE strings
        return randomized_smile_orderings

    def randomize_smiles(self, mol):
        '''Returns a random (dearomatized) SMILES given an rdkit mol object of a molecule.
        Parameters:
        mol (rdkit.Chem.rdchem.Mol) :  RdKit mol object (None if invalid smile string smi)

        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object  (None if invalid smile string smi)
        '''
        if not mol:
            return None

        Chem.Kekulize(mol)
        return rdkit.Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False,  kekuleSmiles=True) 

    def sanitize_smiles(self, smi):
        '''Return a canonical smile representation of smi

        Parameters:
        smi (string) : smile string to be canonicalized 

        Returns:
        mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
        smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
        conversion_successful (bool): True/False to indicate if conversion was  successful 
        '''
        try:
            mol = smi2mol(smi, sanitize=True)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            return (mol, smi_canon, True)
        except:
            return (None, None, False)

    def get_selfie_chars(self, selfie):
        '''Obtain a list of all selfie characters in string selfie

        Parameters: 
        selfie (string) : A selfie string - representing a molecule 

        Example: 
        >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
        ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

        Returns:
        chars_selfie: list of selfie characters present in molecule selfie
        '''
        chars_selfie = [] # A list of all SELFIE sybols from string selfie
        while selfie != '':
            chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
            selfie = selfie[selfie.find(']')+1:]
        return chars_selfie

















