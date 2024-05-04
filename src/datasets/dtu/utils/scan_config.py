
def load_scans(config_dir, stage):
    file_name = f'{config_dir}/{stage.lower()}.txt'
    with open(file_name, encoding='utf-8') as stage_scans_config:
        scan_ids = [line.rstrip() for line in stage_scans_config]

    return scan_ids
