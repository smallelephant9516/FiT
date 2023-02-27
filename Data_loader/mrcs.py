import os
import struct
from collections import OrderedDict
from typing import Any, Optional, Tuple
import numpy as np

DTYPE_FOR_MODE = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: "2h",  # complex number from 2 shorts
    4: np.complex64,
    6: np.uint16,
    12: np.float16,
    16: "3B",
}  # RBG values
MODE_FOR_DTYPE = {vv: kk for kk, vv in DTYPE_FOR_MODE.items()}


class MRCHeader:
    """MRC header class"""

    FIELDS = [
        "nx",
        "ny",
        "nz",  # int
        "mode",  # int
        "nxstart",
        "nystart",
        "nzstart",  # int
        "mx",
        "my",
        "mz",  # int
        "xlen",
        "ylen",
        "zlen",  # float
        "alpha",
        "beta",
        "gamma",  # float
        "mapc",
        "mapr",
        "maps",  # int
        "amin",
        "amax",
        "amean",  # float
        "ispg",
        "next",
        "creatid",  # int, int, short, [pad 30]
        "nint",
        "nreal",  # short, [pad 20]
        "imodStamp",
        "imodFlags",  # int
        "idtype",
        "lens",
        "nd1",
        "nd2",
        "vd1",
        "vd2",  # short
        "tilt_ox",
        "tilt_oy",
        "tilt_oz",  # float
        "tilt_cx",
        "tilt_cy",
        "tilt_cz",  # float
        "xorg",
        "yorg",
        "zorg",  # float
        "cmap",
        "stamp",
        "rms",  # char[4], float
        "nlabl",
        "labels",
    ]  # int, char[10][80]
    FSTR = "3ii3i3i3f3f3i3f2ih30x2h20x2i6h6f3f4s4sfi800s"
    STRUCT = struct.Struct(FSTR)

    def __init__(self, header_values, extended_header=b""):
        self.fields = OrderedDict(zip(self.FIELDS, header_values))
        self.extended_header = extended_header
        self.D = self.fields["nx"]
        self.dtype = DTYPE_FOR_MODE[self.fields["mode"]]

    def __str__(self):
        return f"Header: {self.fields}\nExtended header: {self.extended_header}"

    @classmethod
    def parse(cls, fname):
        with open(fname, "rb") as f:
            header = cls(cls.STRUCT.unpack(f.read(1024)))
            extbytes = header.fields["next"]
            extended_header = f.read(extbytes)
            header.extended_header = extended_header
        return header

    @classmethod
    def make_default_header(
        cls, data, is_vol=True, Apix=1.0, xorg=0.0, yorg=0.0, zorg=0.0
    ):
        nz, ny, nx = data.shape
        ispg = 1 if is_vol else 0
        if is_vol:
            dmin, dmax, dmean, rms = data.min(), data.max(), data.mean(), data.std()
        else:  # use undefined values for image stacks
            dmin, dmax, dmean, rms = -1, -2, -3, -1
        vals = [
            nx,
            ny,
            nz,
            2,  # mode = 2 for 32-bit float
            0,
            0,
            0,  # nxstart, nystart, nzstart
            nx,
            ny,
            nz,  # mx, my, mz
            Apix * nx,
            Apix * ny,
            Apix * nz,  # cella
            90.0,
            90.0,
            90.0,  # cellb
            1,
            2,
            3,  # mapc, mapr, maps
            dmin,
            dmax,
            dmean,
            ispg,
            0,  # exthd_size
            0,  # creatid
            0,
            0,  # nint, nreal
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            xorg,
            yorg,
            zorg,
            b"MAP " if is_vol else b"\x00" * 4,
            b"\x00" * 4,  # cmap, stamp
            rms,  # rms
            0,  # nlabl
            b"\x00" * 800,  # labels
        ]
        return cls(vals)

    def write(self, fh):
        buf = self.STRUCT.pack(*list(self.fields.values()))
        fh.write(buf)
        fh.write(self.extended_header)

    def get_apix(self):
        return self.fields["xlen"] / self.fields["nx"]

    def update_apix(self, Apix):
        self.fields["xlen"] = self.fields["nx"] * Apix
        self.fields["ylen"] = self.fields["ny"] * Apix
        self.fields["zlen"] = self.fields["nz"] * Apix

    def get_origin(self):
        return self.fields["xorg"], self.fields["yorg"], self.fields["zorg"]

    def update_origin(self, xorg, yorg, zorg):
        self.fields["xorg"] = xorg
        self.fields["yorg"] = yorg
        self.fields["zorg"] = zorg


def parse_header(fname: str) -> MRCHeader:
    return MRCHeader.parse(fname)

class LazyImage:
    """On-the-fly image loading"""

    def __init__(self, fname: str, shape: Tuple[int, int], dtype: Any, offset: int):
        self.fname = fname
        self.shape = shape
        self.dtype = dtype
        self.offset = offset

    def get(self) -> np.ndarray:
        with open(self.fname) as f:
            f.seek(self.offset)
            image = np.fromfile(
                f, dtype=self.dtype, count=np.product(self.shape)
            ).reshape(self.shape)
        return image
