#! /usr/bin/env python3

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import NamedTuple

import pandas as pd


class TransfacMotifExample(ABC):
    @classmethod
    @abstractmethod
    def lines_header(cls) -> list[str]: ...

    @classmethod
    @abstractmethod
    def lines_matrix(cls) -> list[str]: ...

    @classmethod
    @abstractmethod
    def lines_footer(cls) -> list[str]: ...

    @classmethod
    @abstractmethod
    def key_value_separator(cls) -> str: ...

    @classmethod
    @abstractmethod
    def matrix_value_content_separator(cls) -> str: ...

    @classmethod
    @abstractmethod
    def matrix_key_value_separator(cls) -> str: ...

    @classmethod
    @abstractmethod
    def consensus(cls) -> str | None: ...

    @classmethod
    @property
    def lines_motif(cls) -> list[str]:
        return cls.lines_header + cls.lines_matrix + cls.lines_footer

    @classmethod
    @property
    def matrix_size(cls) -> int:
        return len(cls.lines_matrix) - 1  # -1 for the header

    @classmethod
    @property
    def alphabet(cls) -> tuple[str, ...]:
        header_matrix_cols = (
            cls.lines_matrix[0]
            .strip("\n")
            .split(cls.matrix_key_value_separator, maxsplit=1)[1]
            .split(cls.matrix_value_content_separator)
        )
        return tuple([element for element in header_matrix_cols if element != ""])

    @classmethod
    def _get_lines_matrix(cls, N: int = 3) -> list[str]:
        if not N > 0:
            N = 1
        if N >= len(cls.lines_matrix):
            N = len(cls.lines_matrix)
        return cls.lines_matrix[: N + 1]  # +1 for the header

    @classmethod
    def get_lines_matrix_for_test(cls, N: int = 3) -> list[tuple[str, NamedTuple]]:
        lines = cls._get_lines_matrix(N)
        _Expected = namedtuple("_Expected", ["key", "values", "consensus"])
        expected = []
        for line in lines:
            key, content = line.strip("\n").split(cls.matrix_key_value_separator, maxsplit=1)
            content = content.split(cls.matrix_value_content_separator)
            content = [element for element in content if element != ""]
            if key != "P0":
                if len(content) == len(cls.alphabet):
                    values = tuple(content)
                    consensus = None
                else:
                    values = tuple(content[:-1])
                    consensus = content[-1]
            else:
                values = tuple(content)
                consensus = None
            expected.append(_Expected(key, values, consensus))

        return list(zip(lines, expected))

    @classmethod
    def get_matrix_as_dataframe(cls) -> pd.DataFrame:
        data = [
            expected_parsed_elements
            for (line, expected_parsed_elements) in cls.get_lines_matrix_for_test(N=cls.matrix_size)
        ]
        df = pd.DataFrame([parsed.values for parsed in data[1:]])
        df.columns = data[0].values
        df.index = [parsed.key for parsed in data[1:]]
        return df

    @classmethod
    @property
    def matrix(cls) -> pd.DataFrame:
        return cls.get_matrix_as_dataframe()


# CORRECT MOTIFS
# ==============


class MCrossTransfacMotif(TransfacMotifExample):
    # From HepG2.AGGF1.top10.cluster.m1.00.mat

    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return [
            "AC\tHepG2.AGGF1.0\n",
            "XX\n",
            "TY\tMotif\n",
            "XX\n",
            "ID\tHepG2.AGGF1.0\n",
            "XX\n",
            "NA\tHepG2.AGGF1.0\n",
            "XX\n",
            "DE\tN=555, Consensus=NNGAAGAAANN,NNGAAGATANN, Score=146.374017578534\n",
        ]

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return [
            "P0\tA\tC\tG\tT\n",
            "01\t232\t67\t102\t154\tN\n",
            "02\t170\t18\t175\t192\tN\n",
            "03\t45\t16\t464\t30\tG\n",
            "04\t499\t15\t33\t8\tA\n",
            "05\t504\t0\t31\t20\tA\n",
            "06\t37\t7\t498\t13\tG\n",
            "07\t485\t18\t36\t16\tA\n",
            "08\t326\t10\t20\t199\tA\n",
            "09\t400\t24\t72\t59\tA\n",
            "10\t174\t77\t141\t163\tN\n",
            "11\t145\t69\t150\t191\tN\n",
        ]

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return [
            "XX\n",
            "IC\t0.139 0.258 1.111 1.387 1.465 1.389 1.262 0.737 0.734 0.055 0.076\n",
            "XX\n",
            "XL\n",
            "00\t40\n",
            "01\t80\n",
            "02\t88\n",
            "03\t20\n",
            "04\t13\n",
            "05\t19\n",
            "06\t11\n",
            "07\t62\n",
            "08\t41\n",
            "09\t74\n",
            "10\t107\n",
            "XX\n",
            "//\n",
        ]

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def consensus(cls) -> str:
        return "NNGAAGAAANN"


class RsatTransfacMotif(TransfacMotifExample):
    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return [
            "AC  assembly_1\n",
            "XX\n",
            "ID  assembly_1\n",
            "XX\n",
            "DE  wdkTCACGTGAmhw\n",
        ]

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return [
            "P0           a         c         g         t\n",
            "1           13         2         4         7\n",
            "2            8         2         7         9\n",
            "3            4         0        12        10\n",
            "4            1         1         5        19\n",
            "5            0        26         0         0\n",
            "6           26         0         0         0\n",
            "7            0        24         1         1\n",
            "8            1         1        24         0\n",
            "9            0         0         0        26\n",
            "10           0         0        26         0\n",
            "11          19         5         1         1\n",
            "12          10        12         0         4\n",
            "13           9         7         2         8\n",
            "14           7         4         2        13\n",
        ]

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return [
            "XX\n",
            "CC  program: feature\n",
            "CC  matrix.nb: 1\n",
            "CC  matrix.nb: 1\n",
            "CC  sites: 26\n",
            "CC  consensus.strict: atgTCACGTGAcat\n",
            "CC  consensus.strict.rc: ATGTCACGTGACAT\n",
            "CC  consensus.IUPAC: wdkTCACGTGAmhw\n",
            "CC  consensus.IUPAC.rc: WDKTCACGTGAMHW\n",
            "CC  consensus.regexp: [at][agt][gt]TCACGTGA[ac][act][at]\n",
            "CC  consensus.regexp.rc: [AT][AGT][GT]TCACGTGA[AC][ACT][AT]\n",
            "XX\n",
            "//\n",
        ]

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return "  "

    # NOTE: poor formatting from RSAT output — the parser filters empty fields to handle this.

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return " "

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return " "

    @classmethod
    @property
    def consensus(cls) -> None:
        return None


class GenericMinimalTransfacMotif(TransfacMotifExample):
    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return [
            "ID  generic_minimal_motif\n",
            "XX\n",
        ]

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return [
            "P0  A C G T\n",
            "01  0.6 0.1 0.1 0.2\n",
            "02  0.1 0.6 0.1 0.2\n",
            "03  0.1 0.1 0.6 0.2\n",
            "04  0.1 0.1 0.2 0.6\n",
            "05  0.25 0.25 0.25 0.25\n",
            "06  0.25 0.25 0.0 0.5\n",
            "07  1.0 0.0 0.0 0.0\n",
        ]

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return [
            "XX\n",
            "//\n",
        ]

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return "  "

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return "  "

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return " "

    @classmethod
    @property
    def consensus(cls) -> None:
        return None


class JasparTransfacMotif(TransfacMotifExample):
    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return [
            "AC MA0004.1\n",
            "XX\n",
            "ID Arnt\n",
            "XX\n",
            "DE MA0004.1 Arnt ; From JASPAR\n",
        ]

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return [
            "PO	A	C	G	T\n",
            "01	4.0	16.0	0.0	0.0\n",
            "02	19.0	0.0	1.0	0.0\n",
            "03	0.0	20.0	0.0	0.0\n",
            "04	0.0	0.0	20.0	0.0\n",
            "05	0.0	0.0	0.0	20.0\n",
            "06	0.0	0.0	20.0	0.0\n",
        ]

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return [
            "XX\n",
            "CC tax_group:vertebrates\n",
            "CC tf_family:PAS domain factors\n",
            "CC tf_class:Basic helix-loop-helix factors (bHLH)\n",
            "CC pubmed_ids:7592839\n",
            "CC uniprot_ids:P53762\n",
            "CC data_type:SELEX\n",
            "XX\n",
            "//\n",
        ]

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return " "

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def consensus(cls) -> None:
        return None


# INCORRECT MOTIFS
# ================


class MalformedTransfacMotif(TransfacMotifExample):
    # Found in file K562.SERBP1.top10.cluster.m1.06.mat
    # Matrix file stops right after the P0 header line — XX and // are missing.

    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return [
            "AC\tK562.SERBP1.7\n",
            "XX\n",
            "TY\tMotif\n",
            "XX\n",
            "ID\tK562.SERBP1.7\n",
            "XX\n",
            "NA\tK562.SERBP1.7\n",
            "XX\n",
            "DE\tN=0, Consensus=NNGCATATCNN, Score=0\n",
        ]

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return ["P0\tA\tC\tG\tT\n"]

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return []

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return "\t"

    @classmethod
    @property
    def consensus(cls) -> None:
        return None


class EmptyTransfacMotif(TransfacMotifExample):
    @classmethod
    @property
    def lines_header(cls) -> list[str]:
        return []

    @classmethod
    @property
    def lines_matrix(cls) -> list[str]:
        return []

    @classmethod
    @property
    def lines_footer(cls) -> list[str]:
        return []

    @classmethod
    @property
    def key_value_separator(cls) -> str:
        return ""

    @classmethod
    @property
    def matrix_key_value_separator(cls) -> str:
        return ""

    @classmethod
    @property
    def matrix_value_content_separator(cls) -> str:
        return ""

    @classmethod
    @property
    def consensus(cls) -> None:
        return None
