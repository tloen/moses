import rdkit.Chem as Chem
import torch
import torch.nn as nn

from .nnutils import index_select_nd
from collections import OrderedDict, defaultdict

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si',
             'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
             'Fe', 'Al', 'I', 'B', 'K', 'Se',
             'Zn', 'H', 'Cu', 'Mn', 'unknown']  # TODO

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1  # TODO
BOND_FDIM = 5  # TODO
MAX_NB = 10  # TODO


class JTMPN(nn.Module):

    @staticmethod
    def _device(model):
        return next(model.parameters()).device

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, cand_batch, tree_mess):
        device = JTMPN._device(self)
        fatoms, fbonds = [], []
        in_bonds, all_bonds = [], []
        mess_dict, all_mess = OrderedDict(), [torch.zeros(self.hidden_size, device=device)]
        total_atoms = 0
        scope = []

        for e, vec in tree_mess.items():
            mess_dict[e] = len(all_mess) # index into all_mess
            all_mess.append(vec) # holds messages

        bonds_between_nodes = defaultdict(list) #[defaultdict(list) for c in cand_batch]

        for mol, all_nodes, ctr_node in cand_batch:
            n_atoms = mol.GetNumAtoms()

            for atom in mol.GetAtoms(): # compile atom features in fatoms
                fatoms.append(atom_features(atom, device))
                in_bonds.append([]) # set up length of in_bonds

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms # note that indices are shifted by atom total
                y = a2.GetIdx() + total_atoms

                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1 # which node?
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1 # which node?

                bfeature = bond_features(bond, device)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((x, y))
                b1 = len(all_bonds) - 1
                fbonds.append(torch.cat([fatoms[x], bfeature], 0))
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y, x))
                b2 = len(all_bonds) - 1
                fbonds.append(torch.cat([fatoms[y], bfeature], 0))
                in_bonds[x].append(b)

                if 0 <= x_bid != y_bid >= 0: # if x and y are in different nodes
                    if (x_bid, y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid, y_bid)]
                        in_bonds[y].append(mess_idx) # message vector into y
                        bonds_between_nodes[(x_bid, y_bid)].append(b1)
                    if (y_bid, x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid, x_bid)]
                        in_bonds[x].append(mess_idx) # message vector into x
                        bonds_between_nodes[(y_bid, x_bid)].append(b2)

            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0).to(device=device) if len(fbonds) != 0 else torch.zeros([0, 40]).to(device=device)
        agraph = torch.zeros(total_atoms, MAX_NB, dtype=torch.long, device=device)
        bgraph = torch.zeros(total_bonds, MAX_NB, dtype=torch.long, device=device)
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        # bgraph[bond index, i] = i-th 
        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]): # connections from other nodes: i, i-th index into all_mess
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y: # ignore y unless  
                    bgraph[b1, i] = b2

        binput = self.W_i(fbonds)
        graph_message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0)
            nei_message = index_select_nd(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = nn.ReLU()(binput + nei_message)

        collated = {
          k: graph_message.index_select(
            0,
            torch.Tensor(v).to(device=device).long()
          ).mean(0) for k, v in bonds_between_nodes.items()
        }
        import pdb; pdb.set_trace(); 
        # print(collated.keys())

        message = torch.cat([tree_message, graph_message], dim=0)
        nei_message = index_select_nd(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs, collated


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, device):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
                        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
                        + [atom.GetIsAromatic()]).to(device=device)


def bond_features(bond, device):
    bt = bond.GetBondType()
    return torch.Tensor(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]).to(device=device)
