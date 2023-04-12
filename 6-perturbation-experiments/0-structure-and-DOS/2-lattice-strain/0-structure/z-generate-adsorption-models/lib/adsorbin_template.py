

adsorbin_template = """\
##################+++ AdsorbKit settings +++##################

################+++ Adsorb Settings +++################
adsorb_style          =  top     # 吸附模式：top(hcp/fcc)/bridge/centre/manual
    adsorb_distance   =  2          # 吸附分子与基体间原子最小距离(键长)
    
substrate      =  POSCAR_substrate        # 基体文件
    substrate_ref     =  1,2        # 基体的参考点, 或手动设定坐标
adsorbate      =  POSCAR_adsorbate        # 含吸附分子的文件
    adsorbate_select  =  all        # 算作吸附分子的原子      
    adsorbate_ref     =  1          # 吸附分子的参考点(选2个则为中点) 

auto_offset           =  True       # 吸附分子与基体最小距离超出(键长 ± 0.1 Å)范围时将吸附分子延Z轴偏移

################+++ Output Settings +++################
output_filename       =  POSCAR     # 输出文件名
coordinate            =  Cartesian  # 坐标系
selective_dynamics    =  False       # 选择性优化
    atoms_to_release  =  3-5        # 放开的原子(按先substrate、后adsorbate排序)
"""
