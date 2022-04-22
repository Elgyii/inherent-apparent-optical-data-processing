import sys
from pathlib import Path

from step1_data_import import step1_data_import
from step2_data_select import step2_data_select
from step3_data_interpolate import step3_data_interpolate
from step4_data_calibrate import step4_data_calibrate
from step5_Rrs_estimate import step5_rrs_estimate
from step6_save_Rrs import step6_save_rrs

if __name__ == '__main__':
    CRUISE = '202204'  # 202109
    HOME_DIR = Path().home().joinpath(
        fr'Documents\NPEC\Toyama\Hayatsuki\{CRUISE}')
    DATA_PATH = HOME_DIR.joinpath(r'OpticalData\RAMSES')

    try:
        for path in DATA_PATH.iterdir():
            step1_data_import(data_path=DATA_PATH)
            step2_data_select(data_path=DATA_PATH)
            step3_data_interpolate(data_path=DATA_PATH)
            step4_data_calibrate(data_path=DATA_PATH)
            step5_rrs_estimate(data_path=DATA_PATH)
        step6_save_rrs(data_path=DATA_PATH)

    except KeyboardInterrupt:
        sys.exit(0)
