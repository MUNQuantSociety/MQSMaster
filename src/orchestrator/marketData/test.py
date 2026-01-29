"""
Financial Modeling Prep API Data Fetcher

This script provides a step-by-step guide on how to pull historical and intraday stock data using the Financial Modeling Prep API.
Read the official documentation for more thorough steps.

Copy this into a jupyter notebook if you have the functionality, it'll be easier!

### Instructions:
0. Create a venv.

1. Install dependencies:
   ```sh
   pip install --no-cache-dir --only-binary :all: -r requirements.txt
   ```
   Ensure you are in the root director while running this.

2. Run the script:
   ```
   python data_infra/marketData/testScript.py
   ```

This script fetches:
- Historical stock data over a specified date range.
- Intraday stock data with user-defined intervals.
"""

import pandas as pd
import requests
from datetime import datetime

import os
from dotenv import load_dotenv
load_dotenv()
from common.auth.apiAuth import APIAuth

def get_historical_data(tickers, from_date, to_date, api_key):
    """
    Fetch historical stock data from Financial Modeling Prep API.

    Args:
        tickers (list or str): A single ticker as a string or multiple tickers as a list (e.g., "AAPL" or ["AAPL", "MSFT"]).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str): Your API key from Financial Modeling Prep.

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data.
    """
    if isinstance(tickers, list):
        tickers = ",".join(tickers)
    
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{tickers}?from={from_date}&to={to_date}&apikey={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        historical_data = []
        
        if 'historical' in data:
            return pd.DataFrame(data['historical'])
        elif 'historicalStockList' in data:
            for stock in data['historicalStockList']:
                for record in stock['historical']:
                    record['ticker'] = stock['symbol']
                    historical_data.append(record)
            return pd.DataFrame(historical_data)
        else:
            print("No historical data found.")
            return pd.DataFrame()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()


def get_intraday_data(tickers, from_date, to_date, interval, api_key):
    """
    Fetch intraday historical stock data from Financial Modeling Prep API.

    Args:
        tickers (list or str): Ticker(s) as a string or list (e.g., "AAPL" or ["AAPL", "MSFT"]).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (int): Interval in minutes (1, 5, 15, 30, 60).
        api_key (str): Your API key.

    Returns:
        pd.DataFrame: A DataFrame containing intraday stock data.
    """
    if isinstance(tickers, list):
        tickers = ",".join(tickers)
    
    interval_map = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "1hour"}
    interval_str = interval_map.get(interval, "5min")
    
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval_str}/{tickers}?from={from_date}&to={to_date}&apikey={api_key}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()
    

def get_realtime_quote(tickers, api_key, exchange="NASDAQ"):
    """
    Fetch realtime stock data from Financial Modeling Prep API.

    Args:
        tickers (list or str): Ticker(s) as a string or list (e.g., "AAPL" or ["AAPL", "MSFT"]).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (int): Interval in minutes (1, 5, 15, 30, 60).
        api_key (str): Your API key.

    Returns:
        pd.DataFrame: A DataFrame containing intraday stock data.
    """
    if isinstance(tickers, list):
        tickers = ",".join(tickers)

    url = f"https://financialmodelingprep.com/api/v3/quote-short/{tickers}?exchange={exchange}&apikey={api_key}"

    response = requests.get(url)
    
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()

def get_realtime_data(api_key, exchange="NASDAQ"):
    """
    Fetch real-time 1-minute interval stock data from Financial Modeling Prep API for the entire exchange and return the latest available record.

    Args:
        api_key (str): Your API key.
        exchange (str): Stock exchange (default: "NASDAQ").

    Returns:
        dict: A dictionary containing the latest available stock data, or None if no data is available.
    """

    url = "https://financialmodelingprep.com/stable/batch-exchange-quote?"
    params = {
        "short": "false",
        "exchange": exchange,
        "apikey": api_key
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()
        
    return data[-3:-1]

if __name__ == "__main__":
    # User Inputs
    selected_tickers = ["^VIX"]
    from_date = "2025-03-06"
    to_date = "2025-03-07"
    interval = 5  # Interval in minutes (1, 5, 15, 30, 60)
    api= APIAuth()
    api_key = api.get_fmp_api_key()  # Find this in data_infra/authentication/apiAuth.py
    ""
    # Fetch historical data
    print("Fetching historical data...")
    historical_df = get_historical_data(selected_tickers, from_date, to_date, api_key)
    if not historical_df.empty:
        print(historical_df.head())
    else:
        print("No historical data available.")
    
    # Fetch intraday data
    print("Fetching intraday data...")
    intraday_df = get_intraday_data(selected_tickers, from_date, to_date, interval, api_key)
    if not intraday_df.empty:
        print(intraday_df.describe())
        print("------")
        print(intraday_df.head(5))
    else:
        print("No intraday data available.")
    
    print("Data fetching complete.")

    MY_TICKERS = ['^VIX'] #'MMM', 'ETNB', 'ATEN', 'AAON', 'AIR', 'ABT', 'ABBV', 'ANF', 'ABM', 'ASO', 'ACHC', 'ACAD', 'ACIW', 'ACMR', 'AYI', 'TIC', 'GOLF', 'ACVA', 'AHCO', 'ADPT', 'ADUS', 'ADEA', 'ADNT', 'ADMA', 'ADBE', 'ADT', 'ATGE', 'AAP', 'AEIS', 'AMD', 'ACM', 'AVAV', 'AGCO', 'A', 'AGL', 'AGYS', 'AGIO', 'ATSG', 'AKAM', 'AKRO', 'ALG', 'ALRM', 'ALK', 'AIN', 'ALIT', 'ALGN', 'ALHC', 'ALKT', 'ALGT', 'ALGM', 'ALSN', 'ALNY', 'ATEC', 'ALTR', 'AMZN', 'AMBA', 'AMC', 'DOX', 'AMED', 'AMTM', 'AAL', 'AEO', 'AME', 'AMGN', 'FOLD', 'AMKR', 'AMRX', 'AMPH', 'APH', 'AMPL', 'ADI', 'ANIP', 'ANSS', 'AM', 'AR', 'APA', 'APLS', 'APG', 'APGE', 'APPF', 'APPN', 'AAPL', 'AIT', 'AMAT', 'AAOI', 'ARMK', 'ARCB', 'ACLX', 'ACHR', 'AROC', 'ACA', 'RCUS', 'ARQT', 'ARDX', 'ARDT', 'AGX', 'ARHS', 'ARIS', 'ANET', 'ARLO', 'AWI', 'ARW', 'ARWR', 'SPRY', 'AORT', 'ARVN', 'ASAN', 'ABG', 'ASGN', 'AZPN', 'ALAB', 'ASTH', 'ATKR', 'BATRA', 'AESI', 'ATMU', 'ATRC', 'AUR', 'ADSK', 'ADP', 'AN', 'AZO', 'AVTR', 'AVPT', 'RNA', 'AVDX', 'CAR', 'AVT', 'ACLS', 'AXON', 'AXSM', 'AZEK', 'AZTA', 'AZZ', 'BMI', 'BKR', 'BBSI', 'BBWI', 'BAX', 'BECN', 'BEAM', 'BDX', 'ONC', 'BELFA', 'BDC', 'BLTE', 'BHE', 'BSY', 'BBY', 'BBAI', 'BILL', 'BIO', 'TECH', 'BCRX', 'BIIB', 'BHVN', 'BLFS', 'BMRN', 'BKV', 'BSM', 'BLKB', 'BL', 'BE', 'BLBD', 'BPMC', 'BA', 'BOOT', 'BAH', 'BWA', 'BSX', 'BOX', 'BYD', 'BRC', 'BRZE', 'BBIO', 'BFAM', 'BTSG', 'BV', 'BCO', 'EAT', 'BMY', 'VTOL', 'AVGO', 'BKD', 'BRKR', 'BC', 'BKE', 'BLDR', 'BURL', 'BWXT', 'CHRW', 'AI', 'CACI', 'WHD', 'CDNS', 'CDRE', 'CZR', 'CRC', 'CALX', 'CWH', 'CAH', 'CDNA', 'KMX', 'CCL', 'CARR', 'CRI', 'CVNA', 'CWST', 'CPRX', 'CAT', 'CAVA', 'CVCO', 'CBZ', 'CCCS', 'CDW', 'CLDX', 'COR', 'CNGO', 'CNC', 'CNTA', 'CTRI', 'CCS', 'CERT', 'CGON', 'SKY', 'CHX', 'CRL', 'GTLS', 'CAKE', 'CHE', 'LNG', 'CQP', 'CVX', 'CHWY', 'CMG', 'CHH', 'CHRD', 'CHDN', 'CIEN', 'CNK', 'CTAS', 'CRUS', 'CSCO', 'CIVI', 'CLH', 'YOU', 'CWAN', 'NET', 'CLOV', 'CNX', 'CGNX', 'CTSH', 'COHR', 'COLM', 'FIX', 'COMM', 'CVLT', 'CRK', 'CON', 'CNXC', 'CFLT', 'CNMD', 'COP', 'ROAD', 'COO', 'CPRT', 'CORT', 'CNM', 'CXW', 'GLW', 'CRSR', 'CRVL', 'CTRA', 'CPNG', 'COUR', 'CRAI', 'CBRL', 'CR', 'CXT', 'CRDO', 'CRGY', 'CRCT', 'CRNX', 'CROX', 'CRWD', 'CSGS', 'CSWI', 'CSX', 'CTS', 'CMI', 'CW', 'CTOS', 'CVI', 'CVS', 'CYTK', 'DAN', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DKL', 'DK', 'DELL', 'DAL', 'DNLI', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DKS', 'DBD', 'DGII', 'DOCN', 'DDS', 'DIOD', 'IRON', 'DSGR', 'DNOW', 'DOCU', 'DLB', 'DPZ', 'DCI', 'DFIN', 'DMLP', 'DORM', 'DV', 'DOV', 'DOCS', 'DHI', 'DKNG', 'DFH', 'DRVN', 'DBX', 'DTM', 'DUOL', 'BROS', 'DXC', 'DXPE', 'DY', 'DT', 'DVAX', 'DYN', 'ETN', 'EBAY', 'EWTX', 'EW', 'ELAN', 'ESTC', 'ELV', 'LLY', 'EME', 'EMR', 'EHC', 'NDOI', 'ET', 'EPAC', 'ENS', 'ENFN', 'ELVN', 'ENOV', 'ENVX', 'ENPH', 'NPO', 'ENSG', 'ENTG', 'EPD', 'NVST', 'EOG', 'EPAM', 'PLUS', 'EQT', 'ESAB', 'ESE', 'ETSY', 'EVEX', 'EVCM', 'EVRI', 'ECG', 'EVH', 'EXAS', 'EE', 'EXEL', 'EXLS', 'EXE', 'EXPD', 'EXPO', 'XPRO', 'EXTR', 'XOM', 'FFIV', 'FAST', 'FSS', 'FDX', 'FERG', 'FA', 'FSLR', 'FWRG', 'FCFS', 'FIVN', 'FLEX', 'FND', 'FLOC', 'FLS', 'FLNC', 'FLR', 'FLUT', 'FL', 'F', 'FOR', 'FORM', 'FTNT', 'FTV', 'FTRE', 'FBIN', 'FOXF', 'FELE', 'FRPT', 'FRSH', 'FTDR', 'ULCC', 'FCN', 'GIII', 'GME', 'GAP', 'IT', 'GTES', 'GEHC', 'GEV', 'GEN', 'WGS', 'GNRC', 'GD', 'GE', 'GM', 'GEL', 'G', 'GNTX', 'THRM', 'GPC', 'GEO', 'GERN', 'ROCK', 'GILD', 'GTLB', 'GKOS', 'GBTG', 'GLP', 'GFS', 'GMED', 'GMS', 'GDRX', 'GT', 'GRC', 'GGG', 'GHC', 'GRAL', 'LOPE', 'GVA', 'GRBK', 'GBX', 'GDYN', 'GFF', 'GPI', 'GH', 'GRDN', 'GWRE', 'GPOR', 'GXO', 'GYRE', 'HEES', 'HRB', 'HAE', 'HAL', 'HALO', 'HBI', 'HOG', 'HLIT', 'HRMY', 'HROW', 'HAS', 'HAYW', 'HCA', 'HQY', 'HEI', 'HLIO', 'HLX', 'HP', 'HSIC', 'HRI', 'HTZ', 'HES', 'HESM', 'HPE', 'HXL', 'DINO', 'HPK', 'HI', 'HLMN', 'HGV', 'HLT', 'HNI', 'HOLX', 'HD', 'HON', 'HWM', 'HPQ', 'HUBG', 'HUBB', 'HUBS', 'HUM', 'JBHT', 'HII', 'HURN', 'H', 'IBTA', 'ICFI', 'ICUI', 'IDYA', 'IEX', 'IDXX', 'IESC', 'ITW', 'ILMN', 'IBRX', 'IMVT', 'PI', 'INCY', 'INFN', 'INR', 'INFA', 'IR', 'INGM', 'INOD', 'INVX', 'INVA', 'NSIT', 'INSM', 'NSP', 'INSP', 'IBP', 'PODD', 'INTA', 'ITGR', 'IART', 'IAS', 'INTC', 'NTLA', 'IDCC', 'TILE', 'IGT', 'INSW', 'IBM', 'ITCI', 'INTU', 'LUNR', 'ISRG', 'IONS', 'IONQ', 'IOVA', 'IPGP', 'IQV', 'IRTC', 'ITRI', 'ITT', 'JBL', 'J', 'JAMF', 'JBI', 'JANX', 'JBTM', 'JBLU', 'FROG', 'JOBY', 'JNJ', 'JCI', 'JNPR', 'KAI', 'KRMN', 'KBH', 'KBR', 'KMT', 'KEYS', 'KRP', 'KMI', 'KLC', 'KNTK', 'KNSA', 'KEX', 'KLAC', 'KVYO', 'KNX', 'KN', 'KGS', 'KSS', 'KTB', 'KFY', 'KOS', 'KTOS', 'KRYS', 'KYMR', 'KD', 'LHX', 'LZB', 'LH', 'LRCX', 'LB', 'LSTR', 'LNTH', 'LVS', 'LSCC', 'LAUR', 'LCII', 'LEA', 'LZ', 'LEGN', 'LEG', 'LDOS', 'LMAT', 'LEN', 'LII', 'DRS', 'LEVI', 'LGIH', 'LBRT', 'LTH', 'LFST', 'LGND', 'LNW', 'LECO', 'LNN', 'LQDA', 'LQDT', 'LAD', 'LFUS', 'LYV', 'RAMP', 'LKQ', 'LOAR', 'LMT', 'LOW', 'LCID', 'LUCK', 'LITE', 'MHO', 'MNR', 'MTSI', 'M', 'MSGE', 'MSGS', 'MDGL', 'MGY', 'MANH', 'MNKD', 'MAN', 'MPC', 'MAR', 'VAC', 'MRTN', 'MRVL', 'MAS', 'MASI', 'MTZ', 'MBC', 'MTDR', 'MATX', 'MAT', 'MTTR', 'MMS', 'MXL', 'MCD', 'MCK', 'MEDP', 'MRK', 'MRCY', 'MLNK', 'MMSI', 'MTH', 'MTSR', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MIDD', 'MLKN', 'MRP', 'MDXG', 'MIR', 'MIRM', 'MCW', 'MKSI', 'MRNA', 'MOD', 'MHK', 'MOH', 'MCRI', 'MDB', 'MPWR', 'MOG/A', 'MSI', 'MPLX', 'MRC', 'MSA', 'MSM', 'MLI', 'MWA', 'MUR', 'MUSA', 'MYRG', 'NABL', 'NNE', 'NTRA', 'NHC', 'EYE', 'NVGS', 'NCNO', 'NATL', 'VYX', 'NEOG', 'NEO', 'NPWR', 'NTAP', 'NTCT', 'NBIX', 'NFE', 'NWL', 'NEXT', 'NN', 'NXT', 'NIKA', 'NKE', 'NE', 'NDSN', 'JWN', 'NSC', 'NOG', 'NOC', 'NCLH', 'NOV', 'NOVT', 'NVAX', 'NRIX', 'SMR', 'NTNX', 'NUVL', 'NVEE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'OII', 'OCUL', 'OKTA', 'ODFL', 'OLO', 'OMCL', 'ON', 'OKE', 'OS', 'ONTO', 'KAR', 'OPK', 'OPCH', 'ORCL', 'OGN', 'OSCR', 'OSK', 'OSIS', 'OTIS', 'OVV', 'PCAR', 'PCRX', 'PACS', 'PD', 'PLTR', 'PANW', 'PZZA', 'PAR', 'FNA', 'PH', 'PSN', 'PATK', 'PDCO', 'PTEN', 'PAYX', 'PAYC', 'PYCR', 'PCTY', 'PBF', 'CNXN', 'MD', 'PEGA', 'PTON', 'PENG', 'PENN', 'PAG', 'PEN', 'PRDO', 'PR', 'PFE', 'PSX', 'PHIN', 'PLAB', 'PHR', 'PBI', 'PAA', 'PAGP', 'PLNT', 'PL', 'PLYA', 'PLXS', 'PLUG', 'PII', 'POOL', 'PTLO', 'POWL', 'POWI', 'PINC', 'PRIM', 'PRVA', 'PRCT', 'PCOR', 'ACDC', 'PRG', 'PRGS', 'PGNY', 'PRO', 'PTGX', 'PTC', 'PTCT', 'PLSE', 'PHM', 'PSTG', 'PCT', 'PVH', 'QTWO', 'QRVO', 'QCOM', 'QLYS', 'PWR', 'QS', 'DGX', 'QDEL', 'QXO', 'RDNT', 'RL', 'RMBS', 'RRC', 'RPD', 'RBC', 'RXRX', 'RRR', 'RRX', 'REGN', 'RGEN', 'RSG', 'REZI', 'RMD', 'REVG', 'RVMD', 'RVLV', 'RVTY', 'RH', 'RYTM', 'RGTI', 'RNG', 'RIVN', 'RHI', 'RKLB', 'RCKT', 'ROK', 'ROIV', 'ROL', 'ROP', 'ROST', 'RCL', 'RPRX', 'RES', 'RTX', 'RBRK', 'RUSHA', 'RSI', 'RXO', 'RXST', 'R', 'SOC', 'SABR', 'SAIA', 'SAIL', 'CRM', 'IOT', 'SNDK', 'SANM', 'SRPT', 'SVV', 'SLB', 'SNDR', 'SRRK', 'SDGR', 'SAIC', 'SMG', 'STX', 'SEM', 'WTTR', 'SEMR', 'SMTC', 'ST', 'S', 'SCI', 'NOW', 'TTAN', 'SHAK', 'SN', 'SIG', 'SLAB', 'SITE', 'SITM', 'STR', 'FUN', 'SKX', 'SKYW', 'SWKS', 'SM', 'AOS', 'SDHC', 'SNA', 'SNOW', 'SEI', 'SWI', 'SLNO', 'SOLV', 'SGI', 'SAH', 'SONO', 'SHC', 'SOUN', 'LUV', 'SPR', 'SWTX', 'CXM', 'SPT', 'SPSC', 'SPXC', 'SYRE', 'SSII', 'SSNC', 'SARO', 'SXI', 'SWK', 'SBUX', 'SCS', 'STE', 'STRL', 'SHOO', 'STRA', 'LRN', 'GPCR', 'SYK', 'SMMT', 'SUN', 'RUN', 'SMCI', 'SUPN', 'SGRY', 'SG', 'SYM', 'SYNA', 'SNDX', 'SNPS', 'TALO', 'TNDM', 'TPR', 'TRGP', 'TARS', 'TASK', 'TMHC', 'SNX', 'TDOC', 'TDY', 'TFX', 'TEM', 'TENB', 'THC', 'TNC', 'TDC', 'TER', 'TEX', 'TSLA', 'TTEK', 'TXN', 'TPL', 'TXRH', 'TXT', 'TGTX', 'CI', 'TMO', 'THO', 'TDW', 'TKR', 'TJX', 'TKO', 'TOL', 'BLD', 'MODG', 'TTC', 'TSCO', 'TDG', 'TMDX', 'TNL', 'TVTX', 'TPH', 'TRMB', 'TNET', 'TRN', 'TGI', 'TTMI', 'TPC', 'TWLO', 'TWST', 'TYL', 'UHAL', 'USPH', 'UI', 'UDMY', 'UFPT', 'PATH', 'ULS', 'ULTA', 'UCTT', 'RARE', 'UAA', 'UNF', 'UNP', 'UAL', 'UPS', 'PRKS', 'URI', 'UTHR', 'UNH', 'U', 'OLED', 'UHS', 'UTI', 'UPBD', 'URBN', 'USAC', 'VVX', 'MTN', 'VAL', 'VLO', 'VMI', 'VVV', 'VRNS', 'PCVX', 'VECO', 'VEEV', 'VG', 'VERA', 'VCYT', 'VLTO', 'VCEL', 'VRNT', 'VRRM', 'VERX', 'VRTX', 'VRT', 'VSTS', 'VFC', 'DSP', 'VSAT', 'VTRS', 'VIAV', 'VICR', 'VSCO', 'VKTX', 'VNOM', 'VIR', 'VRDN', 'VSH', 'VC', 'VTLE', 'VNT', 'VSEC', 'WAB', 'WRBY', 'WM', 'WAT', 'WSO', 'WTS', 'WVE', 'W', 'WAY', 'WFRD', 'WEN', 'WERN', 'WCC', 'WST', 'WDC', 'WES', 'WHR', 'WLY', 'WMB', 'WSM', 'WSC', 'WING', 'WINA', 'WGO', 'WWW', 'WWD', 'WDAY', 'WK', 'WKC', 'WOR', 'GWW', 'WH', 'WYNN', 'XNCR', 'XMTR', 'XPO', 'XYL', 'YETI', 'YUM', 'ZBRA', 'ZETA', 'ZBH', 'ZTS', 'ZM', 'ZI', 'ZS', 'ZWS']
    
    out = get_realtime_quote(MY_TICKERS, api_key, exchange="INDEX")
    out2 = get_historical_data(['^VIX'], '2025-03-01', '2025-03-07', api_key)
    out3 = get_intraday_data(['^VIX'], '2025-03-06', '2025-03-07', 5, api_key)
    print(out)
    print(out2)
    print(out3)

