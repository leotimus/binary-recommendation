from pathlib import Path
import logging, gdown, os

logger = logging.getLogger("DataFetcher")
logging.basicConfig()

def downloadData(url='https://drive.google.com/uc?id=1Xjse6RzUI34yD1TB7HkTYRhp_uIrBfjf',
                 destDir="data"):
  # prepare destination
  dataFile = 'lastfm_play.csv'
  dest = Path(destDir)
  dest.parent.mkdir(parents=True, exist_ok=True)

  # download zip
  path = Path(destDir, dataFile)
  if not path.exists():
    logger.info("downloading file: %s.", dataFile)
    gdown.download(url)
    os.rename(dataFile, path.absolute())
