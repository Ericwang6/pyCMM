import os
from typing import Dict
import xml
import xml.etree.ElementTree as ET
import parmed

from ..units import HARTREE2KJ, BOHR2NM


def prettify_xml(xmlstr: str):
    pretxml = xml.dom.minidom.parseString(xmlstr)
    pretstr = pretxml.toprettyxml()
    pretstr = '\n'.join([x for x in pretstr.split('\n')[1:] if x.strip()])
    return pretstr


def format_attribs(attrs):
    f_attrs = {}
    for key, val in attrs.items():
        if isinstance(val, float):
            f_attrs[key] = f'{val:.8e}' if val != 0.0 else '0.0'
        elif val is None:
            raise Exception("Value is None")
        else:
            f_attrs[key] = str(val)
    return f_attrs


class BondedAtoms:
    def __init__(self, *atoms):
        self.atoms = tuple(atoms)
        self.atoms_reversed = tuple(reversed(atoms))

    def __hash__(self):
        return min(hash(self.atoms), hash(self.atoms_reversed))
    
    def __eq__(self, other):
        return self.atoms == other.atoms or self.atoms == other.atoms_reversed
    
    def __repr__(self):
        return self.atoms.__repr__()
    
    def __getitem__(self, i):
        return self.atoms[i]
    


def convert_qforce_to_cmm(
    pdb_path: os.PathLike,
    atype_defs: Dict[str, str],
    qforce_xml: os.PathLike,
    input_cmm_xml: os.PathLike,
    output_cmm_xml: os.PathLike = ''
):
    
    ff = ET.parse(input_cmm_xml)
    root = ff.getroot()


    pdb = parmed.load_file(pdb_path)
    atypes = [atype_defs[at.name] for at in pdb.atoms]
    qforce_ff = ET.parse(qforce_xml)

    bond_attrs = {}
    angle_attrs = {}
    angle_angle_attrs = {}
    torsion_attrs = {}
    torsion_angle_attrs = {}
    torsion_angle_angle_attrs = {}
    torsion_bond_attrs = {}

    for force in qforce_ff.findall("Forces/Force"):
        if force.attrib['name'] == 'Bond':
            for bo in force.findall("Bonds/Bond"):
                atype1 = atypes[int(bo.attrib['p1'])]
                atype2 = atypes[int(bo.attrib['p2'])]
                if (atype1, atype2) in bond_attrs or (atype2, atype1) in bond_attrs:
                    continue
                r0 = float(bo.attrib['param1']) / BOHR2NM
                D = float(bo.attrib['param3']) / HARTREE2KJ
                kb = float(bo.attrib['param2']) * BOHR2NM * BOHR2NM / HARTREE2KJ
                bond = {"type1": atype1, "type2": atype2, "r_eq": r0, "D": D, "k_b": kb}
                bond_attrs[(atype1, atype2)] = bond
        elif force.attrib['name'] == 'Angle':
            for ang in force.findall("Angles/Angle"):
                atype1, atype2, atype3 = atypes[int(ang.attrib['p1'])], atypes[int(ang.attrib['p2'])], atypes[int(ang.attrib['p3'])]
                if (atype1, atype2, atype3) in angle_attrs or (atype3, atype2, atype1) in angle_attrs:
                    continue
                teq = float(ang.attrib['param1']) 
                kth = float(ang.attrib['param2']) / HARTREE2KJ
                angle_attrs[(atype1, atype2, atype3)] = {
                    'type1': atype1, 'type2': atype2, 'type3': atype3,
                    'theta_eq': teq, 'k_theta': kth
                }
        elif force.attrib['name'] == 'BondBond':
            for bb in force.findall("Bonds/Bond"):
                p1, p2, p3, p4 = map(int, [bb.attrib[f'p{i+1}'] for i in range(4)])
                i, j, k = None, None, None
                if p1 == p3:
                    i, j, k = p2, p1, p4
                elif p1 == p4:
                    i, j, k = p2, p1, p3
                elif p2 == p3:
                    i, j, k = p1, p2, p4
                elif p2 == p4:
                    i, j, k = p1, p2, p3
                else:
                    raise Exception("Bonds are not share same atom")
                req1 = float(bb.attrib['param1']) / BOHR2NM
                req2 = float(bb.attrib['param2']) / BOHR2NM
                kbb = float(bb.attrib['param3']) * BOHR2NM * BOHR2NM / HARTREE2KJ
                if (atypes[i], atypes[j], atypes[k]) in angle_attrs:
                    angle_attrs[(atypes[i], atypes[j], atypes[k])]['r_eq_1'] = req1
                    angle_attrs[(atypes[i], atypes[j], atypes[k])]['r_eq_2'] = req2
                    angle_attrs[(atypes[i], atypes[j], atypes[k])]['k_bb'] = kbb
                elif (atypes[k], atypes[j], atypes[i]) in angle_attrs:
                    angle_attrs[(atypes[k], atypes[j], atypes[i])]['r_eq_1'] = req2
                    angle_attrs[(atypes[k], atypes[j], atypes[i])]['k_bb'] = kbb
                else:
                    raise Exception("No angles found")
        elif force.attrib['name'] == 'BondAngle':
            for ba in force.findall("Bonds/Bond"):
                p1, p2, p3, p4, p5 = map(int, [ba.attrib[f'p{i+1}'] for i in range(5)])
                kba = float(ba.attrib['param3']) * BOHR2NM / HARTREE2KJ
                bond = (p4, p5) if p2 == p4 else (p5, p4)
                if (atypes[p1], atypes[p2], atypes[p3]) in angle_attrs:
                    attr = 'k_ba_1' if bond[1] == p1 else 'k_ba_2'
                    angle_attrs[(atypes[p1], atypes[p2], atypes[p3])][attr] = kba
                else:
                    attr = 'k_ba_2' if bond[1] == p1 else 'k_ba_1'
                    angle_attrs[(atypes[p1], atypes[p2], atypes[p3])][attr] = kba
        elif force.attrib['name'] == 'AngleAngle':
            for aa in force.findall("Bonds/Bond"):
                p1, p2, p3, p4, p5, p6 = map(int, [aa.attrib[f'p{i+1}'] for i in range(6)])
                ctype = 1 if p2 == p5 else 2
                theq1 = float(aa.attrib['param1'])
                theq2 = float(aa.attrib['param2'])
                kaa = float(aa.attrib['param3']) / HARTREE2KJ
                aatypes = [atypes[p] for p in [p1, p2, p3, p4, p5, p6]]

                trials  = [
                    [atypes[p] for p in [p1, p2, p3, p4, p5, p6]],
                    [atypes[p] for p in [p3, p2, p1, p4, p5, p6]],
                    [atypes[p] for p in [p1, p2, p3, p6, p5, p4]],
                    [atypes[p] for p in [p3, p2, p1, p6, p5, p4]]
                ]
                if any(tuple(t + [ctype]) in angle_angle_attrs for t in trials):
                    continue

                attrs = {f'type{i+1}': atype for i, atype in enumerate(aatypes)}
                attrs['ctype'] = ctype
                attrs['theta_eq_1'] = theq1
                attrs['theta_eq_2'] = theq2
                attrs['k_aa'] = kaa
                angle_angle_attrs[tuple(aatypes + [ctype])] = attrs
        elif force.attrib['name'] == 'DihedralAngle' and force.attrib['particles'] == '7':
            for ta in force.findall("Bonds/Bond"):
                p1, p2, p3, p4, p5, p6, p7 = map(int, [ta.attrib[f'p{i+1}'] for i in range(7)])
                k = float(ta.attrib['param1']) / HARTREE2KJ
                theta0 = float(ta.attrib['param2'])
                n = int(float(ta.attrib['param3']))
                phi0 = float(ta.attrib['param4'])

                torsion_types = [atypes[p1], atypes[p2], atypes[p3], atypes[p4]]
                angle_types = [atypes[p5], atypes[p6], atypes[p7]]

                ctype = None
                if BondedAtoms(p5, p6, p7) == BondedAtoms(p1, p2, p3) or BondedAtoms(p5, p6, p7) == BondedAtoms(p2, p3, p4):
                    ctype = 1
                elif BondedAtoms(p2, p3) == BondedAtoms(p5, p6) or BondedAtoms(p2, p3) == BondedAtoms(p6, p7):
                    ctype = 2
                elif BondedAtoms(p1, p2) == BondedAtoms(p5, p6) or BondedAtoms(p1, p2) == BondedAtoms(p6, p7) or BondedAtoms(p3, p4) == BondedAtoms(p5, p6) or BondedAtoms(p3, p4) == BondedAtoms(p6, p7):
                    ctype = 3
                else:
                    raise Exception("Unknown torsion angle couple type")
                
                key = [atypes[p] for p in [p1, p2, p3, p4, p5, p6, p7]] + [ctype]
                trials = [
                    key,
                    key[:4][::-1] + key[4:],
                    key[:4] + key[4:-1][::-1] + [key[-1]],
                    key[:4][::-1] + key[4:-1][::-1] + [key[-1]]
                ]
                for trial in trials:
                    t = tuple(trial)
                    if t in torsion_angle_attrs:
                        key = t
                        break
                else:
                    key = tuple(key)
                
                attrs = torsion_angle_attrs.get(key, [])
                for param in attrs:
                    if param[0] == n:
                        break
                else:
                    attrs.append((n, theta0, k, phi0))
                
                torsion_angle_attrs[key] = attrs
        elif force.attrib['name'] == 'DihedralBond':
            for tb in force.findall("Bonds/Bond"):
                p1, p2, p3, p4, p5, p6  = map(int, [tb.attrib[f'p{i+1}'] for i in range(6)])
                k = float(tb.attrib['param1']) / HARTREE2KJ * BOHR2NM
                r0 = float(tb.attrib['param2']) / BOHR2NM
                n = int(float(tb.attrib['param3']))
                phi0 = float(tb.attrib['param4'])

                ctype = None
                if BondedAtoms(p2, p3) == BondedAtoms(p5, p6):
                    ctype = 1
                elif BondedAtoms(p1, p2) == BondedAtoms(p5, p6) or BondedAtoms(p3, p4) == BondedAtoms(p5, p6):
                    ctype = 2
                elif p5 == p2 or p5 == p3 or p6 == p2 or p6 == p3:
                    ctype = 3
                else:
                    raise Exception("Unknown torsion bond couple type")
                

                key = [atypes[p] for p in [p1, p2, p3, p4, p5, p6]] + [ctype]
                trials = [
                    key,
                    key[:4][::-1] + key[4:],
                    key[:4] + key[4:-1][::-1] + [key[-1]],
                    key[:4][::-1] + key[4:-1][::-1] + [key[-1]]
                ]
                for trial in trials:
                    t = tuple(trial)
                    if t in torsion_bond_attrs:
                        key = t
                        break
                else:
                    key = tuple(key)
                
                attrs = torsion_bond_attrs.get(key, [])
                for param in attrs:
                    if param[0] == n:
                        break
                else:
                    attrs.append((n, r0, k, phi0))
                
                torsion_bond_attrs[key] = attrs
        elif force.attrib['name'] == 'DihedralAngle' and force.attrib['particles'] == '4':
            for taa in force.findall("Bonds/Bond"):
                p1, p2, p3, p4 = map(int, [taa.attrib[f'p{i+1}'] for i in range(4)])
                torsion_types = [atypes[p] for p in [p1, p2, p3, p4]]
                key = BondedAtoms(*torsion_types)
                if key in torsion_angle_angle_attrs:
                    continue
                attr = torsion_angle_angle_attrs.get(key, [])
                attr.append((
                    int(float(taa.attrib['param4'])),
                    float(taa.attrib['param1']) / HARTREE2KJ,
                    float(taa.attrib['param2']),
                    float(taa.attrib['param3'])
                ))
                torsion_angle_angle_attrs[key] = attr
        elif force.attrib['name'] == 'PeriodicDihedral':
            for tor in force.findall("Torsions/Torsion"):
                p1, p2, p3, p4 = map(int, [tor.attrib[f'p{i+1}'] for i in range(4)])
                torsion_types = [atypes[p] for p in [p1, p2, p3, p4]]
                key = BondedAtoms(*torsion_types)

                n = int(float(tor.attrib['param2']))
                k = float(tor.attrib['param1']) / HARTREE2KJ
                phi0 = float(tor.attrib['param3'])

                attrs = torsion_attrs.get(key, [])
                for param in attrs:
                    if param[0] == n:
                        break
                else:
                    attrs.append((n, k, phi0))
                torsion_attrs[key] = attrs


    for key in torsion_angle_attrs:
        assert len(torsion_angle_attrs[key]) == 4
        torsion_angle_attrs[key].sort(key=lambda x: x[0])
        attr = {f'type{i+1}': key[i] for i in range(7)}
        attr['ctype'] = key[-1]
        attr['theta_eq'] = torsion_angle_attrs[key][0][1]
        for i in range(4):
            attr[f'per{i+1}'] = torsion_angle_attrs[key][i][0]
            attr[f'phase{i+1}'] = torsion_angle_attrs[key][i][-1]
            attr[f'k_ta_{i+1}'] = torsion_angle_attrs[key][i][-2]
        torsion_angle_attrs[key] = attr


    for key in torsion_bond_attrs:
        assert len(torsion_bond_attrs[key]) == 4
        torsion_bond_attrs[key].sort(key=lambda x: x[0])
        attr = {f'type{i+1}': key[i] for i in range(6)}
        attr['ctype'] = key[-1]
        attr['r_eq'] = torsion_bond_attrs[key][0][1]
        for i in range(4):
            attr[f'per{i+1}'] = torsion_bond_attrs[key][i][0]
            attr[f'phase{i+1}'] = torsion_bond_attrs[key][i][-1]
            attr[f'k_tb_{i+1}'] = torsion_bond_attrs[key][i][-2]
        torsion_bond_attrs[key] = attr


    for key in torsion_attrs:
        assert len(torsion_attrs[key]) == 4
        torsion_attrs[key].sort(key=lambda x: x[0])
        attr = {f'type{i+1}': key[i] for i in range(4)}
        for i in range(4):
            attr[f'per{i+1}'] = torsion_attrs[key][i][0]
            attr[f'phase{i+1}'] = torsion_attrs[key][i][2]
            attr[f'k{i+1}'] = torsion_attrs[key][i][1]
        attr['theta_eq_1'] = None
        attr['theta_eq_2'] = None
        attr['k_taa_1'] = 0.0
        attr['k_taa_2'] = 0.0
        attr['k_taa_3'] = 0.0
        attr['k_taa_4'] = 0.0
        torsion_attrs[key] = attr


    for key in torsion_angle_angle_attrs:
        for attrs in torsion_angle_angle_attrs[key]:
            for i in range(4):
                if torsion_attrs[key][f'per{i+1}'] == attrs[0]:
                    torsion_attrs[key]['theta_eq_1'] = attrs[-2]
                    torsion_attrs[key]['theta_eq_2'] = attrs[-1]
                    torsion_attrs[key][f'k_taa_{i+1}'] = attrs[1]
                    break


    for key in bond_attrs:
        bond_attrs[key]['j_cf'] = 0.0
        bond_attrs[key]['j_cf_pauli'] = 0.0
        
        bond_attrs[key]['k_hardness_b'] = 0.0
        bond_attrs[key]['dip_deriv_1'] = 0.0
        bond_attrs[key]['dip_deriv_2'] = 0.0
        bond_attrs[key]['ct_slope_1'] = 0.0
        bond_attrs[key]['ct_slope_2'] = 0.0


    for key in angle_attrs:
        angle_attrs[key]['j_cf_angle'] = 0.0
        angle_attrs[key]['j_cf_bb'] = 0.0
        
        angle_attrs[key]['k_hardness_angle'] = 0.0
        angle_attrs[key]['k_hardness_bb'] = 0.0


    for attrs in bond_attrs.values():
        ET.SubElement(root.find('Bonds'), 'Bond', format_attribs(attrs))

    for attrs in angle_attrs.values():
        ET.SubElement(root.find('Angles'), 'Angle', format_attribs(attrs))

    for attrs in torsion_attrs.values():
        ET.SubElement(root.find('Torsions'), 'Torsion', format_attribs(attrs))

    for attrs in angle_angle_attrs.values():
        ET.SubElement(root.find('AngleAngleCoupling'), 'AngleAngle', format_attribs(attrs))

    for attrs in torsion_bond_attrs.values():
        ET.SubElement(root.find('TorsionBondCoupling'), 'TorsionBond', format_attribs(attrs))

    for attrs in torsion_angle_attrs.values():
        ET.SubElement(root.find('TorsionAngleCoupling'), 'TorsionAngle', format_attribs(attrs))
    
    outstr = prettify_xml(ET.tostring(root))
    if output_cmm_xml:
        with open(output_cmm_xml, 'w') as f:
            f.write(outstr)
    
    return outstr