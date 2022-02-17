def pretty_name(name):
    if 'OurModel' in name:
        name = name.split('OurModel_')[1]
    if 'FlatMemoryWithSummarization' in name:
        name = name.replace('FlatMemoryWithSummarization', 'Flat')
    if 'Hierarchical_LIFEWATCH_limit_1000_Hierarchical_LIFEWATCH_limit_1000' in name:
        name = name.replace('Hierarchical_LIFEWATCH_limit_1000_Hierarchical_LIFEWATCH_limit_1000', 'HierarchicalLifewatch')
    return name
