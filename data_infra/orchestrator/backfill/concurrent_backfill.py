"""
concurrent_backfill.py
----------------------
A multi-threaded approach for backfilling multiple tickers in parallel
using the existing 'backfill_data' function from 'backfill.py'.
Each ticker is processed in its own thread to reduce total runtime.

Results:
  - Data is injected directly into the 'market_data' table in the database using MQSDBConnector.
"""

import sys
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from psycopg2.extras import execute_values

# Ensure we can import backfill.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from data_infra.orchestrator.backfill.backfill import backfill_data
from data_infra.database.MQSDBConnector import MQSDBConnector

# Number of threads to use. Adjust based on CPU/network constraints.
MAX_WORKERS = 3


def parse_date_arg(date_str):
    """Parses date string in DDMMYY format and returns a datetime.date object."""
    try:
        return datetime.strptime(date_str, "%d%m%y").date()
    except ValueError:
        print(f"❌ Invalid date format: {date_str}. Expected format: DDMMYY (e.g., 040325 for March 4, 2025).")
        sys.exit(1)


def backfill_single_ticker(ticker, start_date, end_date, interval, exchange):
    """
    Calls backfill_data(...) for a single ticker. 
    Instead of writing to CSV, injects data directly into DB.
    """
    try:
        # Fetch the data in-memory (output_filename=None => returns DataFrame)
        df = backfill_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            exchange=exchange,
            output_filename=None
        )

        # Check if data was returned and not empty
        if df is None or df.empty:
            print(f"[{ticker}] No data returned from backfill.")
            return

        # Create DB connection
        db = MQSDBConnector()
        conn = db.get_connection()

        insert_data = []
        for _, row in df.iterrows():
            try:
                insert_data.append((
                    row['ticker'],
                    row['datetime'],  # timestamp
                    row['date'],
                    exchange.lower() if exchange else 'nasdaq',
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(float(row['volume'])),
                ))
            except Exception as parse_ex:
                print(f"[{ticker}] Skipping row due to parsing error: {parse_ex}")
                continue

        # Bulk insert
        insert_sql = """
            INSERT INTO market_data (
                ticker, timestamp, date, exchange,
                open_price, high_price, low_price, close_price, volume
            )
            VALUES %s
        """

        if insert_data:
            with conn.cursor() as cursor:
                execute_values(cursor, insert_sql, insert_data)
            conn.commit()
            print(f"[{ticker}] Inserted {len(insert_data)} rows into DB.")
        else:
            print(f"[{ticker}] No valid rows to insert.")

    except Exception as e:
        print(f"[{ticker}] Error during backfill or insert: {e}")
    finally:
        if 'conn' in locals():
            db.release_connection(conn)


def concurrent_backfill(tickers, start_date, end_date, interval, exchange=None):
    """
    Spawns multiple threads, each calling 'backfill_data' for a single ticker.
    Injects each ticker's results directly into the DB.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    print(f"[ConcurrentBackfill] Starting concurrency for {len(tickers)} tickers.")
    print(f"  Date range: {start_date} to {end_date}, interval={interval} min, exchange={exchange}")
    print(f"  Using up to {MAX_WORKERS} threads.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(backfill_single_ticker, ticker, start_date, end_date, interval, exchange)
            for ticker in tickers
        ]
        for fut in futures:
            try:
                fut.result()
            except Exception as ex:
                print(f"[ConcurrentBackfill:ERROR] A worker failed with: {ex}")

    print("[ConcurrentBackfill] All threads completed.")


if __name__ == "__main__":
    # 1. Define tickers
    MY_TICKERS = ['TXG', 'MMM', 'ETNB', 'ATEN', 'AAON', 'AIR', 'ABT', 'ABBV', 'ANF', 'ABM', 'ASO', 'ACHC', 'ACAD', 'ACIW', 'ACMR', 'AYI', 'TIC', 'GOLF', 'ACVA', 'AHCO', 'ADPT', 'ADUS', 'ADEA', 'ADNT', 'ADMA', 'ADBE', 'ADT', 'ATGE', 'AAP', 'AEIS', 'AMD', 'ACM', 'AVAV', 'AGCO', 'A', 'AGL', 'AGYS', 'AGIO', 'ATSG', 'AKAM', 'AKRO', 'ALG', 'ALRM', 'ALK', 'AIN', 'ALIT', 'ALGN', 'ALHC', 'ALKT', 'ALGT', 'ALGM', 'ALSN', 'ALNY', 'ATEC', 'ALTR', 'AMZN', 'AMBA', 'AMC', 'DOX', 'AMED', 'AMTM', 'AAL', 'AEO', 'AME', 'AMGN', 'FOLD', 'AMKR', 'AMRX', 'AMPH', 'APH', 'AMPL', 'ADI', 'ANIP', 'ANSS', 'AM', 'AR', 'APA', 'APLS', 'APG', 'APGE', 'APPF', 'APPN', 'AAPL', 'AIT', 'AMAT', 'AAOI', 'ARMK', 'ARCB', 'ACLX', 'ACHR', 'AROC', 'ACA', 'RCUS', 'ARQT', 'ARDX', 'ARDT', 'AGX', 'ARHS', 'ARIS', 'ANET', 'ARLO', 'AWI', 'ARW', 'ARWR', 'SPRY', 'AORT', 'ARVN', 'ASAN', 'ABG', 'ASGN', 'AZPN', 'ALAB', 'ASTH', 'ATKR', 'BATRA', 'AESI', 'ATMU', 'ATRC', 'AUR', 'ADSK', 'ADP', 'AN', 'AZO', 'AVTR', 'AVPT', 'RNA', 'AVDX', 'CAR', 'AVT', 'ACLS', 'AXON', 'AXSM', 'AZEK', 'AZTA', 'AZZ', 'BMI', 'BKR', 'BBSI', 'BBWI', 'BAX', 'BECN', 'BEAM', 'BDX', 'ONC', 'BELFA', 'BDC', 'BLTE', 'BHE', 'BSY', 'BBY', 'BBAI', 'BILL', 'BIO', 'TECH', 'BCRX', 'BIIB', 'BHVN', 'BLFS', 'BMRN', 'BKV', 'BSM', 'BLKB', 'BL', 'BE', 'BLBD', 'BPMC', 'BA', 'BOOT', 'BAH', 'BWA', 'BSX', 'BOX', 'BYD', 'BRC', 'BRZE', 'BBIO', 'BFAM', 'BTSG', 'BV', 'BCO', 'EAT', 'BMY', 'VTOL', 'AVGO', 'BKD', 'BRKR', 'BC', 'BKE', 'BLDR', 'BURL', 'BWXT', 'CHRW', 'AI', 'CACI', 'WHD', 'CDNS', 'CDRE', 'CZR', 'CRC', 'CALX', 'CWH', 'CAH', 'CDNA', 'KMX', 'CCL', 'CARR', 'CRI', 'CVNA', 'CWST', 'CPRX', 'CAT', 'CAVA', 'CVCO', 'CBZ', 'CCCS', 'CDW', 'CLDX', 'COR', 'CNGO', 'CNC', 'CNTA', 'CTRI', 'CCS', 'CERT', 'CGON', 'SKY', 'CHX', 'CRL', 'GTLS', 'CAKE', 'CHE', 'LNG', 'CQP', 'CVX', 'CHWY', 'CMG', 'CHH', 'CHRD', 'CHDN', 'CIEN', 'CNK', 'CTAS', 'CRUS', 'CSCO', 'CIVI', 'CLH', 'YOU', 'CWAN', 'NET', 'CLOV', 'CNX', 'CGNX', 'CTSH', 'COHR', 'COLM', 'FIX', 'COMM', 'CVLT', 'CRK', 'CON', 'CNXC', 'CFLT', 'CNMD', 'COP', 'ROAD', 'COO', 'CPRT', 'CORT', 'CNM', 'CXW', 'GLW', 'CRSR', 'CRVL', 'CTRA', 'CPNG', 'COUR', 'CRAI', 'CBRL', 'CR', 'CXT', 'CRDO', 'CRGY', 'CRCT', 'CRNX', 'CROX', 'CRWD', 'CSGS', 'CSWI', 'CSX', 'CTS', 'CMI', 'CW', 'CTOS', 'CVI', 'CVS', 'CYTK', 'DAN', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DKL', 'DK', 'DELL', 'DAL', 'DNLI', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DKS', 'DBD', 'DGII', 'DOCN', 'DDS', 'DIOD', 'IRON', 'DSGR', 'DNOW', 'DOCU', 'DLB', 'DPZ', 'DCI', 'DFIN', 'DMLP', 'DORM', 'DV', 'DOV', 'DOCS', 'DHI', 'DKNG', 'DFH', 'DRVN', 'DBX', 'DTM', 'DUOL', 'BROS', 'DXC', 'DXPE', 'DY', 'DT', 'DVAX', 'DYN', 'ETN', 'EBAY', 'EWTX', 'EW', 'ELAN', 'ESTC', 'ELV', 'LLY', 'EME', 'EMR', 'EHC', 'NDOI', 'ET', 'EPAC', 'ENS', 'ENFN', 'ELVN', 'ENOV', 'ENVX', 'ENPH', 'NPO', 'ENSG', 'ENTG', 'EPD', 'NVST', 'EOG', 'EPAM', 'PLUS', 'EQT', 'ESAB', 'ESE', 'ETSY', 'EVEX', 'EVCM', 'EVRI', 'ECG', 'EVH', 'EXAS', 'EE', 'EXEL', 'EXLS', 'EXE', 'EXPD', 'EXPO', 'XPRO', 'EXTR', 'XOM', 'FFIV', 'FAST', 'FSS', 'FDX', 'FERG', 'FA', 'FSLR', 'FWRG', 'FCFS', 'FIVN', 'FLEX', 'FND', 'FLOC', 'FLS', 'FLNC', 'FLR', 'FLUT', 'FL', 'F', 'FOR', 'FORM', 'FTNT', 'FTV', 'FTRE', 'FBIN', 'FOXF', 'FELE', 'FRPT', 'FRSH', 'FTDR', 'ULCC', 'FCN', 'GIII', 'GME', 'GAP', 'IT', 'GTES', 'GEHC', 'GEV', 'GEN', 'WGS', 'GNRC', 'GD', 'GE', 'GM', 'GEL', 'G', 'GNTX', 'THRM', 'GPC', 'GEO', 'GERN', 'ROCK', 'GILD', 'GTLB', 'GKOS', 'GBTG', 'GLP', 'GFS', 'GMED', 'GMS', 'GDRX', 'GT', 'GRC', 'GGG', 'GHC', 'GRAL', 'LOPE', 'GVA', 'GRBK', 'GBX', 'GDYN', 'GFF', 'GPI', 'GH', 'GRDN', 'GWRE', 'GPOR', 'GXO', 'GYRE', 'HEES', 'HRB', 'HAE', 'HAL', 'HALO', 'HBI', 'HOG', 'HLIT', 'HRMY', 'HROW', 'HAS', 'HAYW', 'HCA', 'HQY', 'HEI', 'HLIO', 'HLX', 'HP', 'HSIC', 'HRI', 'HTZ', 'HES', 'HESM', 'HPE', 'HXL', 'DINO', 'HPK', 'HI', 'HLMN', 'HGV', 'HLT', 'HNI', 'HOLX', 'HD', 'HON', 'HWM', 'HPQ', 'HUBG', 'HUBB', 'HUBS', 'HUM', 'JBHT', 'HII', 'HURN', 'H', 'IBTA', 'ICFI', 'ICUI', 'IDYA', 'IEX', 'IDXX', 'IESC', 'ITW', 'ILMN', 'IBRX', 'IMVT', 'PI', 'INCY', 'INFN', 'INR', 'INFA', 'IR', 'INGM', 'INOD', 'INVX', 'INVA', 'NSIT', 'INSM', 'NSP', 'INSP', 'IBP', 'PODD', 'INTA', 'ITGR', 'IART', 'IAS', 'INTC', 'NTLA', 'IDCC', 'TILE', 'IGT', 'INSW', 'IBM', 'ITCI', 'INTU', 'LUNR', 'ISRG', 'IONS', 'IONQ', 'IOVA', 'IPGP', 'IQV', 'IRTC', 'ITRI', 'ITT', 'JBL', 'J', 'JAMF', 'JBI', 'JANX', 'JBTM', 'JBLU', 'FROG', 'JOBY', 'JNJ', 'JCI', 'JNPR', 'KAI', 'KRMN', 'KBH', 'KBR', 'KMT', 'KEYS', 'KRP', 'KMI', 'KLC', 'KNTK', 'KNSA', 'KEX', 'KLAC', 'KVYO', 'KNX', 'KN', 'KGS', 'KSS', 'KTB', 'KFY', 'KOS', 'KTOS', 'KRYS', 'KYMR', 'KD', 'LHX', 'LZB', 'LH', 'LRCX', 'LB', 'LSTR', 'LNTH', 'LVS', 'LSCC', 'LAUR', 'LCII', 'LEA', 'LZ', 'LEGN', 'LEG', 'LDOS', 'LMAT', 'LEN', 'LII', 'DRS', 'LEVI', 'LGIH', 'LBRT', 'LTH', 'LFST', 'LGND', 'LNW', 'LECO', 'LNN', 'LQDA', 'LQDT', 'LAD', 'LFUS', 'LYV', 'RAMP', 'LKQ', 'LOAR', 'LMT', 'LOW', 'LCID', 'LUCK', 'LITE', 'MHO', 'MNR', 'MTSI', 'M', 'MSGE', 'MSGS', 'MDGL', 'MGY', 'MANH', 'MNKD', 'MAN', 'MPC', 'MAR', 'VAC', 'MRTN', 'MRVL', 'MAS', 'MASI', 'MTZ', 'MBC', 'MTDR', 'MATX', 'MAT', 'MTTR', 'MMS', 'MXL', 'MCD', 'MCK', 'MEDP', 'MRK', 'MRCY', 'MLNK', 'MMSI', 'MTH', 'MTSR', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MIDD', 'MLKN', 'MRP', 'MDXG', 'MIR', 'MIRM', 'MCW', 'MKSI', 'MRNA', 'MOD', 'MHK', 'MOH', 'MCRI', 'MDB', 'MPWR', 'MOG/A', 'MSI', 'MPLX', 'MRC', 'MSA', 'MSM', 'MLI', 'MWA', 'MUR', 'MUSA', 'MYRG', 'NABL', 'NNE', 'NTRA', 'NHC', 'EYE', 'NVGS', 'NCNO', 'NATL', 'VYX', 'NEOG', 'NEO', 'NPWR', 'NTAP', 'NTCT', 'NBIX', 'NFE', 'NWL', 'NEXT', 'NN', 'NXT', 'NIKA', 'NKE', 'NE', 'NDSN', 'JWN', 'NSC', 'NOG', 'NOC', 'NCLH', 'NOV', 'NOVT', 'NVAX', 'NRIX', 'SMR', 'NTNX', 'NUVL', 'NVEE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'OII', 'OCUL', 'OKTA', 'ODFL', 'OLO', 'OMCL', 'ON', 'OKE', 'OS', 'ONTO', 'KAR', 'OPK', 'OPCH', 'ORCL', 'OGN', 'OSCR', 'OSK', 'OSIS', 'OTIS', 'OVV', 'PCAR', 'PCRX', 'PACS', 'PD', 'PLTR', 'PANW', 'PZZA', 'PAR', 'FNA', 'PH', 'PSN', 'PATK', 'PDCO', 'PTEN', 'PAYX', 'PAYC', 'PYCR', 'PCTY', 'PBF', 'CNXN', 'MD', 'PEGA', 'PTON', 'PENG', 'PENN', 'PAG', 'PEN', 'PRDO', 'PR', 'PFE', 'PSX', 'PHIN', 'PLAB', 'PHR', 'PBI', 'PAA', 'PAGP', 'PLNT', 'PL', 'PLYA', 'PLXS', 'PLUG', 'PII', 'POOL', 'PTLO', 'POWL', 'POWI', 'PINC', 'PRIM', 'PRVA', 'PRCT', 'PCOR', 'ACDC', 'PRG', 'PRGS', 'PGNY', 'PRO', 'PTGX', 'PTC', 'PTCT', 'PLSE', 'PHM', 'PSTG', 'PCT', 'PVH', 'QTWO', 'QRVO', 'QCOM', 'QLYS', 'PWR', 'QS', 'DGX', 'QDEL', 'QXO', 'RDNT', 'RL', 'RMBS', 'RRC', 'RPD', 'RBC', 'RXRX', 'RRR', 'RRX', 'REGN', 'RGEN', 'RSG', 'REZI', 'RMD', 'REVG', 'RVMD', 'RVLV', 'RVTY', 'RH', 'RYTM', 'RGTI', 'RNG', 'RIVN', 'RHI', 'RKLB', 'RCKT', 'ROK', 'ROIV', 'ROL', 'ROP', 'ROST', 'RCL', 'RPRX', 'RES', 'RTX', 'RBRK', 'RUSHA', 'RSI', 'RXO', 'RXST', 'R', 'SOC', 'SABR', 'SAIA', 'SAIL', 'CRM', 'IOT', 'SNDK', 'SANM', 'SRPT', 'SVV', 'SLB', 'SNDR', 'SRRK', 'SDGR', 'SAIC', 'SMG', 'STX', 'SEM', 'WTTR', 'SEMR', 'SMTC', 'ST', 'S', 'SCI', 'NOW', 'TTAN', 'SHAK', 'SN', 'SIG', 'SLAB', 'SITE', 'SITM', 'STR', 'FUN', 'SKX', 'SKYW', 'SWKS', 'SM', 'AOS', 'SDHC', 'SNA', 'SNOW', 'SEI', 'SWI', 'SLNO', 'SOLV', 'SGI', 'SAH', 'SONO', 'SHC', 'SOUN', 'LUV', 'SPR', 'SWTX', 'CXM', 'SPT', 'SPSC', 'SPXC', 'SYRE', 'SSII', 'SSNC', 'SARO', 'SXI', 'SWK', 'SBUX', 'SCS', 'STE', 'STRL', 'SHOO', 'STRA', 'LRN', 'GPCR', 'SYK', 'SMMT', 'SUN', 'RUN', 'SMCI', 'SUPN', 'SGRY', 'SG', 'SYM', 'SYNA', 'SNDX', 'SNPS', 'TALO', 'TNDM', 'TPR', 'TRGP', 'TARS', 'TASK', 'TMHC', 'SNX', 'TDOC', 'TDY', 'TFX', 'TEM', 'TENB', 'THC', 'TNC', 'TDC', 'TER', 'TEX', 'TSLA', 'TTEK', 'TXN', 'TPL', 'TXRH', 'TXT', 'TGTX', 'CI', 'TMO', 'THO', 'TDW', 'TKR', 'TJX', 'TKO', 'TOL', 'BLD', 'MODG', 'TTC', 'TSCO', 'TDG', 'TMDX', 'TNL', 'TVTX', 'TPH', 'TRMB', 'TNET', 'TRN', 'TGI', 'TTMI', 'TPC', 'TWLO', 'TWST', 'TYL', 'UHAL', 'USPH', 'UI', 'UDMY', 'UFPT', 'PATH', 'ULS', 'ULTA', 'UCTT', 'RARE', 'UAA', 'UNF', 'UNP', 'UAL', 'UPS', 'PRKS', 'URI', 'UTHR', 'UNH', 'U', 'OLED', 'UHS', 'UTI', 'UPBD', 'URBN', 'USAC', 'VVX', 'MTN', 'VAL', 'VLO', 'VMI', 'VVV', 'VRNS', 'PCVX', 'VECO', 'VEEV', 'VG', 'VERA', 'VCYT', 'VLTO', 'VCEL', 'VRNT', 'VRRM', 'VERX', 'VRTX', 'VRT', 'VSTS', 'VFC', 'DSP', 'VSAT', 'VTRS', 'VIAV', 'VICR', 'VSCO', 'VKTX', 'VNOM', 'VIR', 'VRDN', 'VSH', 'VC', 'VTLE', 'VNT', 'VSEC', 'WAB', 'WRBY', 'WM', 'WAT', 'WSO', 'WTS', 'WVE', 'W', 'WAY', 'WFRD', 'WEN', 'WERN', 'WCC', 'WST', 'WDC', 'WES', 'WHR', 'WLY', 'WMB', 'WSM', 'WSC', 'WING', 'WINA', 'WGO', 'WWW', 'WWD', 'WDAY', 'WK', 'WKC', 'WOR', 'GWW', 'WH', 'WYNN', 'XNCR', 'XMTR', 'XPO', 'XYL', 'YETI', 'YUM', 'ZBRA', 'ZETA', 'ZBH', 'ZTS', 'ZM', 'ZI', 'ZS', 'ZWS']

    # 2. Parse command-line arguments
    start_date_arg = None
    end_date_arg = None

    for arg in sys.argv[1:]:
        if arg.startswith("startdate="):
            start_date_arg = arg.split("=")[1]
        elif arg.startswith("enddate="):
            end_date_arg = arg.split("=")[1]

    if not start_date_arg or not end_date_arg:
        print("❌ Missing required arguments: startdate and enddate.")
        print("Usage: python3 concurrentbackfill.py startdate=DDMMYY enddate=DDMMYY")
        sys.exit(1)

    # Parse date strings
    start_date = parse_date_arg(start_date_arg)
    end_date = parse_date_arg(end_date_arg)

    # 3. Call concurrent backfill
    concurrent_backfill(
        tickers=MY_TICKERS,
        start_date=start_date,
        end_date=end_date,
        interval=1,
        exchange="NASDAQ",
    )

    print("✅ Concurrent backfill completed and data inserted into the database.")