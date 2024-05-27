# 10 materials from ambientCG when searching for 'fabric'.

from pathlib import Path

AMBIENTCG_CLOTH_MATERIALS_DIR = Path(__file__).parent / "assets"


# "Cloth" materials from AbientCG with  a uniform base color
# (as this tends to be visible in normal maps as well)
# https://ambientcg.com/list?category=&date=&createdUsing=&basedOn=&q=cloth&method=&type=Material&sort=Popular
download_links = [
    "https://ambientcg.com/get?file=Fabric030_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric019_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric032_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric023_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric028_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric026_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric022_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric018_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric029_2K-JPG.zip",
    "https://ambientcg.com/get?file=Fabric002_4K-JPG.zip",
]


def download_ambientcg_cloth_materials(target_dir: Path):
    # download and save
    for link in download_links:
        path = target_dir / link.split("=")[-1]
        path
        path = str(path)
        import urllib.request

        req = urllib.request.Request(
            link,
            data=None,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
            },
        )

        f = urllib.request.urlopen(req)
        with open(path, "wb") as fp:
            fp.write(f.read())


def extract_zips(dir: Path):
    import shutil

    # extract zips
    for zip in dir.glob("*.zip"):
        shutil.unpack_archive(str(zip), str(zip.parent / zip.stem))
        zip.unlink()


if __name__ == "__main__":
    target_dir = AMBIENTCG_CLOTH_MATERIALS_DIR
    target_dir.mkdir(exist_ok=True)
    download_ambientcg_cloth_materials(target_dir)
    extract_zips(target_dir)
