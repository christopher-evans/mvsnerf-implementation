
def load_scans(config_dir, stage):
    """
    Load a list of scans from a configuration file.

    :param config_dir: Configuration directory for DTU dataset
    :param stage: Stage, either 'fit', 'validate' or 'test'
    :return: List of scan IDs
    """
    file_name = f'{config_dir}/{stage.lower()}.txt'
    with open(file_name, encoding='utf-8') as stage_scans_config:
        scan_ids = [line.rstrip() for line in stage_scans_config]

    return scan_ids
