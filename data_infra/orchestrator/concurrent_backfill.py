"""
concurrent_backfill.py
----------------------
A multi-threaded approach for backfilling multiple tickers in parallel
using the existing 'backfill_data' function from 'backfill.py'.
Each ticker is processed in its own thread to reduce total runtime.

Results:
  - One CSV file per ticker (named {output_prefix}_{TICKER}.csv)
  - Optionally merges them into one CSV (all_tickers_combined.csv) in a chunked manner
    to avoid excessive memory usage.
"""

import sys
import os
import time
import glob
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Ensure we can import backfill.py from the orchestrator dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_infra.orchestrator.backfill import backfill_data

# Number of threads to use. Adjust based on CPU/network constraints.
MAX_WORKERS = 3

def backfill_single_ticker(ticker, start_date, end_date, interval, exchange, output_filename):
    """
    Calls backfill_data(...) for a single ticker. 
    Writes results to output_filename.
    This function will run in its own thread.
    """
    backfill_data(
        tickers=[ticker],   # pass a list of length 1 
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        exchange=exchange,
        output_filename=output_filename
    )

def concurrent_backfill(
    tickers, 
    start_date, 
    end_date, 
    interval, 
    exchange=None,
    output_prefix="2y_mkt_data"
):
    """
    Spawns multiple threads, each calling 'backfill_data' for a single ticker.
    Writes each ticker's results to its own CSV file:
        {output_prefix}_{TICKER}.csv

    :param tickers: list of ticker symbols (e.g. ["AAPL","MSFT","GOOG"])
    :param start_date: date or str (YYYY-MM-DD)
    :param end_date: date or str
    :param interval: 1,5,15,30,60 for FMP intervals
    :param exchange: optional string, e.g. 'NASDAQ' or 'NYSE'
    :param output_prefix: prefix for the output CSV files
    """
    # Convert input dates if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"[ConcurrentBackfill] Starting concurrency for {len(tickers)} tickers.")
    print(f"  Date range: {start_date} to {end_date}, interval={interval} min, exchange={exchange}")
    print(f"  Using up to {MAX_WORKERS} threads.")
    print(f"  Output prefix: {output_prefix}")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ticker in tickers:
            # Each ticker writes to its own CSV
            csv_name = f"{output_prefix}_{ticker}.csv"
            fut = executor.submit(
                backfill_single_ticker,
                ticker,
                start_date,
                end_date,
                interval,
                exchange,
                csv_name
            )
            futures.append(fut)

        # Optionally wait for all threads to complete
        for fut in futures:
            try:
                fut.result()  # Raises any exception inside the thread
            except Exception as ex:
                print(f"[ConcurrentBackfill:ERROR] A worker failed with: {ex}")

    print("[ConcurrentBackfill] All threads completed.")

if __name__ == "__main__":
    # 1. Define tickers
    MY_TICKERS = ['TXG', 'MMM', 'ETNB', 'ATEN', 'AAON', 'AIR', 'ABT', 'ABBV', 'ANF', 'ABM', 'ASO', 'ACHC', 'ACAD', 'ACIW', 'ACMR', 'AYI', 'TIC', 'GOLF', 'ACVA', 'AHCO', 'ADPT', 'ADUS', 'ADEA', 'ADNT', 'ADMA', 'ADBE', 'ADT', 'ATGE', 'AAP', 'AEIS', 'AMD', 'ACM', 'AVAV', 'AGCO', 'A', 'AGL', 'AGYS', 'AGIO', 'ATSG', 'AKAM', 'AKRO', 'ALG', 'ALRM', 'ALK', 'AIN', 'ALIT', 'ALGN', 'ALHC', 'ALKT', 'ALGT', 'ALGM', 'ALSN', 'ALNY', 'ATEC', 'ALTR', 'AMZN', 'AMBA', 'AMC', 'DOX', 'AMED', 'AMTM', 'AAL', 'AEO', 'AME', 'AMGN', 'FOLD', 'AMKR', 'AMRX', 'AMPH', 'APH', 'AMPL', 'ADI', 'ANIP', 'ANSS', 'AM', 'AR', 'APA', 'APLS', 'APG', 'APGE', 'APPF', 'APPN', 'AAPL', 'AIT', 'AMAT', 'AAOI', 'ARMK', 'ARCB', 'ACLX', 'ACHR', 'AROC', 'ACA', 'RCUS', 'ARQT', 'ARDX', 'ARDT', 'AGX', 'ARHS', 'ARIS', 'ANET', 'ARLO', 'AWI', 'ARW', 'ARWR', 'SPRY', 'AORT', 'ARVN', 'ASAN', 'ABG', 'ASGN', 'AZPN', 'ALAB', 'ASTH', 'ATKR', 'BATRA', 'AESI', 'ATMU', 'ATRC', 'AUR', 'ADSK', 'ADP', 'AN', 'AZO', 'AVTR', 'AVPT', 'RNA', 'AVDX', 'CAR', 'AVT', 'ACLS', 'AXON', 'AXSM', 'AZEK', 'AZTA', 'AZZ', 'BMI', 'BKR', 'BBSI', 'BBWI', 'BAX', 'BECN', 'BEAM', 'BDX', 'ONC', 'BELFA', 'BDC', 'BLTE', 'BHE', 'BSY', 'BBY', 'BBAI', 'BILL', 'BIO', 'TECH', 'BCRX', 'BIIB', 'BHVN', 'BLFS', 'BMRN', 'BKV', 'BSM', 'BLKB', 'BL', 'BE', 'BLBD', 'BPMC', 'BA', 'BOOT', 'BAH', 'BWA', 'BSX', 'BOX', 'BYD', 'BRC', 'BRZE', 'BBIO', 'BFAM', 'BTSG', 'BV', 'BCO', 'EAT', 'BMY', 'VTOL', 'AVGO', 'BKD', 'BRKR', 'BC', 'BKE', 'BLDR', 'BURL', 'BWXT', 'CHRW', 'AI', 'CACI', 'WHD', 'CDNS', 'CDRE', 'CZR', 'CRC', 'CALX', 'CWH', 'CAH', 'CDNA', 'KMX', 'CCL', 'CARR', 'CRI', 'CVNA', 'CWST', 'CPRX', 'CAT', 'CAVA', 'CVCO', 'CBZ', 'CCCS', 'CDW', 'CLDX', 'COR', 'CNGO', 'CNC', 'CNTA', 'CTRI', 'CCS', 'CERT', 'CGON', 'SKY', 'CHX', 'CRL', 'GTLS', 'CAKE', 'CHE', 'LNG', 'CQP', 'CVX', 'CHWY', 'CMG', 'CHH', 'CHRD', 'CHDN', 'CIEN', 'CNK', 'CTAS', 'CRUS', 'CSCO', 'CIVI', 'CLH', 'YOU', 'CWAN', 'NET', 'CLOV', 'CNX', 'CGNX', 'CTSH', 'COHR', 'COLM', 'FIX', 'COMM', 'CVLT', 'CRK', 'CON', 'CNXC', 'CFLT', 'CNMD', 'COP', 'ROAD', 'COO', 'CPRT', 'CORT', 'CNM', 'CXW', 'GLW', 'CRSR', 'CRVL', 'CTRA', 'CPNG', 'COUR', 'CRAI', 'CBRL', 'CR', 'CXT', 'CRDO', 'CRGY', 'CRCT', 'CRNX', 'CROX', 'CRWD', 'CSGS', 'CSWI', 'CSX', 'CTS', 'CMI', 'CW', 'CTOS', 'CVI', 'CVS', 'CYTK', 'DAN', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DKL', 'DK', 'DELL', 'DAL', 'DNLI', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DKS', 'DBD', 'DGII', 'DOCN', 'DDS', 'DIOD', 'IRON', 'DSGR', 'DNOW', 'DOCU', 'DLB', 'DPZ', 'DCI', 'DFIN', 'DMLP', 'DORM', 'DV', 'DOV', 'DOCS', 'DHI', 'DKNG', 'DFH', 'DRVN', 'DBX', 'DTM', 'DUOL', 'BROS', 'DXC', 'DXPE', 'DY', 'DT', 'DVAX', 'DYN', 'ETN', 'EBAY', 'EWTX', 'EW', 'ELAN', 'ESTC', 'ELV', 'LLY', 'EME', 'EMR', 'EHC', 'NDOI', 'ET', 'EPAC', 'ENS', 'ENFN', 'ELVN', 'ENOV', 'ENVX', 'ENPH', 'NPO', 'ENSG', 'ENTG', 'EPD', 'NVST', 'EOG', 'EPAM', 'PLUS', 'EQT', 'ESAB', 'ESE', 'ETSY', 'EVEX', 'EVCM', 'EVRI', 'ECG', 'EVH', 'EXAS', 'EE', 'EXEL', 'EXLS', 'EXE', 'EXPD', 'EXPO', 'XPRO', 'EXTR', 'XOM', 'FFIV', 'FAST', 'FSS', 'FDX', 'FERG', 'FA', 'FSLR', 'FWRG', 'FCFS', 'FIVN', 'FLEX', 'FND', 'FLOC', 'FLS', 'FLNC', 'FLR', 'FLUT', 'FL', 'F', 'FOR', 'FORM', 'FTNT', 'FTV', 'FTRE', 'FBIN', 'FOXF', 'FELE', 'FRPT', 'FRSH', 'FTDR', 'ULCC', 'FCN', 'GIII', 'GME', 'GAP', 'IT', 'GTES', 'GEHC', 'GEV', 'GEN', 'WGS', 'GNRC', 'GD', 'GE', 'GM', 'GEL', 'G', 'GNTX', 'THRM', 'GPC', 'GEO', 'GERN', 'ROCK', 'GILD', 'GTLB', 'GKOS', 'GBTG', 'GLP', 'GFS', 'GMED', 'GMS', 'GDRX', 'GT', 'GRC', 'GGG', 'GHC', 'GRAL', 'LOPE', 'GVA', 'GRBK', 'GBX', 'GDYN', 'GFF', 'GPI', 'GH', 'GRDN', 'GWRE', 'GPOR', 'GXO', 'GYRE', 'HEES', 'HRB', 'HAE', 'HAL', 'HALO', 'HBI', 'HOG', 'HLIT', 'HRMY', 'HROW', 'HAS', 'HAYW', 'HCA', 'HQY', 'HEI', 'HLIO', 'HLX', 'HP', 'HSIC', 'HRI', 'HTZ', 'HES', 'HESM', 'HPE', 'HXL', 'DINO', 'HPK', 'HI', 'HLMN', 'HGV', 'HLT', 'HNI', 'HOLX', 'HD', 'HON', 'HWM', 'HPQ', 'HUBG', 'HUBB', 'HUBS', 'HUM', 'JBHT', 'HII', 'HURN', 'H', 'IBTA', 'ICFI', 'ICUI', 'IDYA', 'IEX', 'IDXX', 'IESC', 'ITW', 'ILMN', 'IBRX', 'IMVT', 'PI', 'INCY', 'INFN', 'INR', 'INFA', 'IR', 'INGM', 'INOD', 'INVX', 'INVA', 'NSIT', 'INSM', 'NSP', 'INSP', 'IBP', 'PODD', 'INTA', 'ITGR', 'IART', 'IAS', 'INTC', 'NTLA', 'IDCC', 'TILE', 'IGT', 'INSW', 'IBM', 'ITCI', 'INTU', 'LUNR', 'ISRG', 'IONS', 'IONQ', 'IOVA', 'IPGP', 'IQV', 'IRTC', 'ITRI', 'ITT', 'JBL', 'J', 'JAMF', 'JBI', 'JANX', 'JBTM', 'JBLU', 'FROG', 'JOBY', 'JNJ', 'JCI', 'JNPR', 'KAI', 'KRMN', 'KBH', 'KBR', 'KMT', 'KEYS', 'KRP', 'KMI', 'KLC', 'KNTK', 'KNSA', 'KEX', 'KLAC', 'KVYO', 'KNX', 'KN', 'KGS', 'KSS', 'KTB', 'KFY', 'KOS', 'KTOS', 'KRYS', 'KYMR', 'KD', 'LHX', 'LZB', 'LH', 'LRCX', 'LB', 'LSTR', 'LNTH', 'LVS', 'LSCC', 'LAUR', 'LCII', 'LEA', 'LZ', 'LEGN', 'LEG', 'LDOS', 'LMAT', 'LEN', 'LII', 'DRS', 'LEVI', 'LGIH', 'LBRT', 'LTH', 'LFST', 'LGND', 'LNW', 'LECO', 'LNN', 'LQDA', 'LQDT', 'LAD', 'LFUS', 'LYV', 'RAMP', 'LKQ', 'LOAR', 'LMT', 'LOW', 'LCID', 'LUCK', 'LITE', 'MHO', 'MNR', 'MTSI', 'M', 'MSGE', 'MSGS', 'MDGL', 'MGY', 'MANH', 'MNKD', 'MAN', 'MPC', 'MAR', 'VAC', 'MRTN', 'MRVL', 'MAS', 'MASI', 'MTZ', 'MBC', 'MTDR', 'MATX', 'MAT', 'MTTR', 'MMS', 'MXL', 'MCD', 'MCK', 'MEDP', 'MRK', 'MRCY', 'MLNK', 'MMSI', 'MTH', 'MTSR', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MIDD', 'MLKN', 'MRP', 'MDXG', 'MIR', 'MIRM', 'MCW', 'MKSI', 'MRNA', 'MOD', 'MHK', 'MOH', 'MCRI', 'MDB', 'MPWR', 'MOG/A', 'MSI', 'MPLX', 'MRC', 'MSA', 'MSM', 'MLI', 'MWA', 'MUR', 'MUSA', 'MYRG', 'NABL', 'NNE', 'NTRA', 'NHC', 'EYE', 'NVGS', 'NCNO', 'NATL', 'VYX', 'NEOG', 'NEO', 'NPWR', 'NTAP', 'NTCT', 'NBIX', 'NFE', 'NWL', 'NEXT', 'NN', 'NXT', 'NIKA', 'NKE', 'NE', 'NDSN', 'JWN', 'NSC', 'NOG', 'NOC', 'NCLH', 'NOV', 'NOVT', 'NVAX', 'NRIX', 'SMR', 'NTNX', 'NUVL', 'NVEE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'OII', 'OCUL', 'OKTA', 'ODFL', 'OLO', 'OMCL', 'ON', 'OKE', 'OS', 'ONTO', 'KAR', 'OPK', 'OPCH', 'ORCL', 'OGN', 'OSCR', 'OSK', 'OSIS', 'OTIS', 'OVV', 'PCAR', 'PCRX', 'PACS', 'PD', 'PLTR', 'PANW', 'PZZA', 'PAR', 'FNA', 'PH', 'PSN', 'PATK', 'PDCO', 'PTEN', 'PAYX', 'PAYC', 'PYCR', 'PCTY', 'PBF', 'CNXN', 'MD', 'PEGA', 'PTON', 'PENG', 'PENN', 'PAG', 'PEN', 'PRDO', 'PR', 'PFE', 'PSX', 'PHIN', 'PLAB', 'PHR', 'PBI', 'PAA', 'PAGP', 'PLNT', 'PL', 'PLYA', 'PLXS', 'PLUG', 'PII', 'POOL', 'PTLO', 'POWL', 'POWI', 'PINC', 'PRIM', 'PRVA', 'PRCT', 'PCOR', 'ACDC', 'PRG', 'PRGS', 'PGNY', 'PRO', 'PTGX', 'PTC', 'PTCT', 'PLSE', 'PHM', 'PSTG', 'PCT', 'PVH', 'QTWO', 'QRVO', 'QCOM', 'QLYS', 'PWR', 'QS', 'DGX', 'QDEL', 'QXO', 'RDNT', 'RL', 'RMBS', 'RRC', 'RPD', 'RBC', 'RXRX', 'RRR', 'RRX', 'REGN', 'RGEN', 'RSG', 'REZI', 'RMD', 'REVG', 'RVMD', 'RVLV', 'RVTY', 'RH', 'RYTM', 'RGTI', 'RNG', 'RIVN', 'RHI', 'RKLB', 'RCKT', 'ROK', 'ROIV', 'ROL', 'ROP', 'ROST', 'RCL', 'RPRX', 'RES', 'RTX', 'RBRK', 'RUSHA', 'RSI', 'RXO', 'RXST', 'R', 'SOC', 'SABR', 'SAIA', 'SAIL', 'CRM', 'IOT', 'SNDK', 'SANM', 'SRPT', 'SVV', 'SLB', 'SNDR', 'SRRK', 'SDGR', 'SAIC', 'SMG', 'STX', 'SEM', 'WTTR', 'SEMR', 'SMTC', 'ST', 'S', 'SCI', 'NOW', 'TTAN', 'SHAK', 'SN', 'SIG', 'SLAB', 'SITE', 'SITM', 'STR', 'FUN', 'SKX', 'SKYW', 'SWKS', 'SM', 'AOS', 'SDHC', 'SNA', 'SNOW', 'SEI', 'SWI', 'SLNO', 'SOLV', 'SGI', 'SAH', 'SONO', 'SHC', 'SOUN', 'LUV', 'SPR', 'SWTX', 'CXM', 'SPT', 'SPSC', 'SPXC', 'SYRE', 'SSII', 'SSNC', 'SARO', 'SXI', 'SWK', 'SBUX', 'SCS', 'STE', 'STRL', 'SHOO', 'STRA', 'LRN', 'GPCR', 'SYK', 'SMMT', 'SUN', 'RUN', 'SMCI', 'SUPN', 'SGRY', 'SG', 'SYM', 'SYNA', 'SNDX', 'SNPS', 'TALO', 'TNDM', 'TPR', 'TRGP', 'TARS', 'TASK', 'TMHC', 'SNX', 'TDOC', 'TDY', 'TFX', 'TEM', 'TENB', 'THC', 'TNC', 'TDC', 'TER', 'TEX', 'TSLA', 'TTEK', 'TXN', 'TPL', 'TXRH', 'TXT', 'TGTX', 'CI', 'TMO', 'THO', 'TDW', 'TKR', 'TJX', 'TKO', 'TOL', 'BLD', 'MODG', 'TTC', 'TSCO', 'TDG', 'TMDX', 'TNL', 'TVTX', 'TPH', 'TRMB', 'TNET', 'TRN', 'TGI', 'TTMI', 'TPC', 'TWLO', 'TWST', 'TYL', 'UHAL', 'USPH', 'UI', 'UDMY', 'UFPT', 'PATH', 'ULS', 'ULTA', 'UCTT', 'RARE', 'UAA', 'UNF', 'UNP', 'UAL', 'UPS', 'PRKS', 'URI', 'UTHR', 'UNH', 'U', 'OLED', 'UHS', 'UTI', 'UPBD', 'URBN', 'USAC', 'VVX', 'MTN', 'VAL', 'VLO', 'VMI', 'VVV', 'VRNS', 'PCVX', 'VECO', 'VEEV', 'VG', 'VERA', 'VCYT', 'VLTO', 'VCEL', 'VRNT', 'VRRM', 'VERX', 'VRTX', 'VRT', 'VSTS', 'VFC', 'DSP', 'VSAT', 'VTRS', 'VIAV', 'VICR', 'VSCO', 'VKTX', 'VNOM', 'VIR', 'VRDN', 'VSH', 'VC', 'VTLE', 'VNT', 'VSEC', 'WAB', 'WRBY', 'WM', 'WAT', 'WSO', 'WTS', 'WVE', 'W', 'WAY', 'WFRD', 'WEN', 'WERN', 'WCC', 'WST', 'WDC', 'WES', 'WHR', 'WLY', 'WMB', 'WSM', 'WSC', 'WING', 'WINA', 'WGO', 'WWW', 'WWD', 'WDAY', 'WK', 'WKC', 'WOR', 'GWW', 'WH', 'WYNN', 'XNCR', 'XMTR', 'XPO', 'XYL', 'YETI', 'YUM', 'ZBRA', 'ZETA', 'ZBH', 'ZTS', 'ZM', 'ZI', 'ZS', 'ZWS']

    # 2. Define date range (2 years of data)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)

    # 3. Call concurrent backfill
    concurrent_backfill(
        tickers=MY_TICKERS,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ",
        output_prefix="2y_mkt_data"
    )

    print("✅ Concurrent backfill completed.")
    print("Note: Each ticker's data is in e.g. 2y_mkt_data_AAPL.csv, 2y_mkt_data_MMM.csv, etc.")

    # 4. Merge CSVs: read all partial CSV files and concatenate chunk by chunk to avoid overloading RAM.
    print("Merging per-ticker CSVs into a single file 'all_tickers_combined.csv' (chunked)...")

    csv_files = glob.glob("2y_mkt_data_*.csv")  # Adjust if needed
    merge_output = "all_tickers_combined.csv"

    # If file exists from a previous run, remove it to start fresh
    if os.path.exists(merge_output):
        os.remove(merge_output)

    header_written = False
    for file in csv_files:
        print(f"  Merging {file} ...")
        for chunk in pd.read_csv(file, chunksize=100_000):
            chunk.to_csv(merge_output, index=False, header=not header_written, mode='a')
            header_written = True

    print("✅ Merging complete. Final file: all_tickers_combined.csv")
