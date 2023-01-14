import hashlib
import numpy as np
import os
import re
import requests
import tarfile
import time
import random

from absl import logging
from typing import List, NoReturn, Tuple


class MMSeqs2Runner:

    r"""Runner object

    Fetches sequence alignment and templates from MMSeqs2 server
    Based on the function run_mmseqs2 from ColabFold (sokrypton/ColabFold)
    Version 62d7558c91a9809712b022faf9d91d8b183c328c

    Relevant publications
    ----------
    * "Clustering huge protein sequence sets in linear time"
      https://doi.org/10.1038/s41467-018-04964-5
    * "MMseqs2 enables sensitive protein sequence searching for the analysis
      of massive data sets"
      https://doi.org/10.1038/nbt.3988

    Private variables
    ----------
    self.job: Job ID (five-char string)
    self.seq: Sequence to search
    self.host_url: URL address to ping for data
    self.t_url: URL address to ping for templates from PDB
    self.n_templates = Number of templates to fetch (default=20)
    self.path: Path to use
    self.tarfile: Compressed file archive to download
    """

    def __init__(
        self,
        job: str,
        seq: str,
        host_url: str = "https://a3m.mmseqs.com",
        t_url: str = "https://a3m-templates.mmseqs.com/template",
        path_suffix: str = "env",
        n_templates: int = 20,
        shuffling_templates: bool = False,
    ):

        r"""Initialize runner object

        Parameters
        ----------
        job : Job name
        seq : Amino acid sequence
        host_url : Website to ping for sequence data
        t_url : Website to ping for template info
        path_suffix : Suffix for path info

        """

        # Clean up sequence
        self.seq = self._cleanseq(seq.upper())

        # Come up with unique job ID for MMSeqs
        self.job = self._define_jobname(job)

        # Save everything else
        self.host_url = host_url
        self.t_url = t_url
        self.n_templates = n_templates
        self.shuffling_templates = shuffling_templates

        self.path = "_".join((self.job, path_suffix))

        if not os.path.isdir(self.path):
            os.system(f"mkdir { self.path }")

        self.tarfile = f"{ self.path }/out.tar.gz"

    def _cleanseq(self, seq) -> str:

        r"""Cleans the sequence to remove whitespace and noncanonical letters

        Parameters
        ----------
        seq : Amino acid sequence (only all 20 here)

        Returns
        ----------
        Cleaned up amin acid sequence

        """

        if any([aa in seq for aa in "BJOUXZ"]):
            logging.warning("Sequence contains non-canonical amino acids!")
            logging.warning("Removing B, J, O, U, X, and Z from sequence")
            seq = re.sub(r"[BJOUXZ]", "", seq)

        return re.sub(r"[^A-Z]", "", "".join(seq.split()))

    def _define_jobname(self, job: str) -> str:

        r"""Provides a unique five-digit identifier for the job name

        Parameters
        ----------
        job : Job name

        Returns
        ----------
        Defined job name

        """

        return "_".join(
            (
                re.sub(r"\W+", "", "".join(job.split())),
                hashlib.sha1(self.seq.encode()).hexdigest()[:5],
            )
        )

    def _submit(self) -> dict:

        r"""Submit job to MMSeqs2 server

        Parameters
        ----------
        None

        Returns
        ----------
        None

        """

        data = {"q": f">101\n{ self.seq }", "mode": "env"}

        res = requests.post(f"{ self.host_url }/ticket/msa", data=data)

        try:
            out = res.json()

        except ValueError:
            out = {"status": "UNKNOWN"}

        return out

    def _status(self, idx: str) -> dict:

        r"""Check status of job

        Parameters
        ----------
        idx : Index assigned by MMSeqs2 server

        Returns
        ----------
        None

        """

        res = requests.get(f"{ self.host_url }/ticket/{ idx }")

        try:
            out = res.json()

        except ValueError:
            out = {"status": "UNKNOWN"}

        return out

    def _download(self, idx: str, path: str) -> NoReturn:

        r"""Download job outputs

        Parameters
        ----------
        idx : Index assigned by MMSeqs2 server
        path : Path to download data

        Returns
        ----------
        None

        """

        res = requests.get(f"{ self.host_url }/result/download/{ idx }")

        with open(path, "wb") as out:
            out.write(res.content)

    def _search_mmseqs2(self) -> NoReturn:

        r"""Run the search and download results
        Heavily modified from ColabFold

        Parameters
        ----------
        None

        Returns
        ----------
        None

        """

        if os.path.isfile(self.tarfile):
            return

        out = self._submit()

        time.sleep(5 + np.random.randint(0, 5))
        while out["status"] in ["UNKNOWN", "RATELIMIT"]:
            # resubmit
            time.sleep(5 + np.random.randint(0, 5))
            out = self._submit()

        logging.debug(f"ID: { out[ 'id' ] }")

        while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
            time.sleep(5 + np.random.randint(0, 5))
            out = self._status(out["id"])

        if out["status"] == "COMPLETE":
            self._download(out["id"], self.tarfile)

        elif out["status"] == "ERROR":
            raise RuntimeError(
                " ".join(
                    (
                        "MMseqs2 API is giving errors.",
                        "Please confirm your input is a valid protein sequence.",
                        "If error persists, please try again in an hour.",
                    )
                )
            )

    def process_templates(self, templates: List[str] = [] ) -> list:

        r"""Process templates and fetch from MMSeqs2 server

        Parameters
        ----------
        templates : list of pdb ids with chain
        exclusion_gpcrs : list of pdb ids without chain

        Returns
        ----------
        Directory containing templates (empty if not using templates)

        """

        path = f"{ self.job }_env/templates_101"
        if os.path.isdir(path):
            os.system(f"rm -r { path }")

        # templates = {}
        logging.info("\t".join(("seq", "pdb", "cid", "evalue")))

        pdbs = []
        check_duplicates = []
        with open(f"{ self.path }/pdb70.m8", "r") as infile:

            for line in infile:
                 
                sl = line.rstrip().split()
                pdb = sl[1]
                pdbid = pdb.split("_")[0]
                if templates:
                    if templates[0] in ["Active", "Inactive", "Intermediate", "G protein", "Arrestin"] and pdbid not in check_duplicates and pdbid not in templates:
                        activation_state = templates[0]
                        url = "http://gpcrdb.org/services/structure/{}".format( pdbid )
                        r = requests.get( url )
                        rj = r.json()
                        if type(rj) is dict and rj["state"] == activation_state:
                            pdbs.append(pdb)
                            check_duplicates.append(pdbid)
                        elif type(rj) is dict and "signalling_protein" in rj:
                            if rj["signalling_protein"]["type"] == activation_state:
                                pdbs.append(pdb)
                                check_duplicates.append(pdbid)    
                                            
                    if len(templates[0]) == 3 and pdbid not in check_duplicates and pdbid not in templates:
                        if templates[0][0] in ["in", "out", "out-like"]:
                            dfg = templates[0][0]
                        elif templates[0][0] == "all":
                            dfg = "all"
                        else:
                            raise RuntimeError("DFG value invalid")
                        if templates[0][1] in ["in", "out",]:    
                            ac_helix = templates[0][1]
                        elif templates[0][1] == "all":
                            ac_helix = "all"
                        else:
                            raise RuntimeError("ac_helix value invalid")       
                        if templates[0][2] in  ["yes", "no", "all"]:
                            salt_bridge = templates[0][2]
                        else:
                            raise RuntimeError("salt_bridge value invalid")
                        url = "https://klifs.net/api_v2/structures_pdb_list?pdb-codes={}".format( pdbid )
                        r = requests.get( url )
                        rj = r.json()
                        #print(rj)
                        if rj[0] != 400:           
                            #take kinase_ID value and search for structure_conformation
                            structure_ID = rj[0]["structure_ID"]
                            url = "https://klifs.net/api_v2/structure_conformation?structure_ID={}".format( structure_ID )
                            r = requests.get( url )
                            rj = r.json()
                            #print(rj)
                            if float(rj[0]["salt_bridge_17_24"]) > 0 and float(rj[0]["salt_bridge_17_24"]) <= 4.5:
                                ref_sb = "yes"
                            else:
                                ref_sb = "no"
                            if dfg != "all" and ac_helix != "all" and salt_bridge != "all":
                                if rj[0]["DFG"] == dfg and rj[0]["ac_helix"] == ac_helix and  salt_bridge == ref_sb:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg != "all" and ac_helix != "all" and salt_bridge == "all":
                                if rj[0]["DFG"] == dfg and rj[0]["ac_helix"] == ac_helix:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg != "all" and ac_helix == "all" and salt_bridge != "all":
                                if rj[0]["DFG"] == dfg and salt_bridge == ref_sb:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg != "all" and ac_helix == "all" and salt_bridge == "all":
                                if rj[0]["DFG"] == dfg:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg == "all" and ac_helix != "all" and salt_bridge != "all":
                                if rj[0]["ac_helix"] == ac_helix and salt_bridge == ref_sb:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg == "all" and ac_helix != "all" and salt_bridge == "all":
                                if rj[0]["ac_helix"] == ac_helix:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg == "all" and ac_helix == "all" and salt_bridge != "all":
                                if salt_bridge == ref_sb:
                                    pdbs.append(pdb)
                                    check_duplicates.append(pdbid)
                            elif dfg == "all" and ac_helix == "all" and salt_bridge == "all":
                                pdbs.append(pdb)
                                check_duplicates.append(pdbid)                           
                    
                    elif pdb in templates or templates[0] in ["Auto", "AUTO", "auto"]:
                        pdbs.append(sl[1])
                        logging.info(f"{ sl[0] }\t{ sl[1] }\t{ sl[2] }\t{ sl[10] }")
        
        #write comma-seprated pdbs to file
        with open(f"{ self.path }/template_pdbs.txt", "w") as outfile:
            for pdb in pdbs:
                outfile.write(f"{ pdb },")
            
        print("PRINTED LIST:", pdbs)
        
        return self.download_templates(pdbs)
        
    def download_templates(self, pdbs) -> str:
        """Shuffle templates."""
        
        path = f"{ self.job }_env/templates_101"
        if os.path.isdir(path):
            os.system(f"rm -r { path }")
            
        if len(pdbs) == 0:
            logging.warning("No templates found.")
            return ""
        else:
            if not os.path.isdir(path):
                os.mkdir(path)
            
            if len(pdbs) > 1 and self.shuffling_templates:
                random.shuffle(pdbs)
            
            pdbs = ",".join(pdbs[: self.n_templates])
        
            logging.info("TEMPLATE PDBS USED: " + pdbs)

            os.system(f"wget -q -O - { self.t_url }/{ pdbs } |tar xzf - -C { path }/")

            os.system(f"cp { path }/pdb70_a3m.ffindex { path }/pdb70_cs219.ffindex")

            os.system(f"touch { path }/pdb70_cs219.ffdata")

            return path

    def _process_alignment(
        self, a3m_files: list, templates: List[str] = []
    ) -> Tuple[str, str]:

        r"""Process sequence alignment
        (modified from ColabFold)

        Parameters
        ----------
        a3m_files : List of files to parse
        token : Token to look for when parsing

        Returns
        ----------
        Tuple with [0] string with alignment, and [1] path to template

        """

        a3m_lines = ""

        for a3m_file in a3m_files:
            for line in open(os.path.join(self.path, a3m_file), "r"):
                if len(line) > 0:
                    a3m_lines += line.replace("\x00", "")

        return a3m_lines, self.process_templates(templates)

    def run_job(self, templates: List[str] = []) -> Tuple[str, str]:

        r"""
        Run sequence alignments using MMseqs2

        Parameters
        ----------
        use_templates: Whether to use templates

        Returns
        ----------
        Tuple with [0] string with alignment, and [1] path to template

        """

        self._search_mmseqs2()

        a3m_files = ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]

        # extract a3m files
        if not os.path.isfile(os.path.join(self.path, a3m_files[0])):
            with tarfile.open(self.tarfile) as tar_gz:
                tar_gz.extractall(self.path)

        return self._process_alignment(a3m_files, templates)
    
    def shuffle_templates(self) -> List:
    
        r"""
        Run sequence alignments using MMseqs2

        Parameters
        ----------
        use_templates: Whether to use templates

        Returns
        ----------
        Tuple with [0] string with alignment, and [1] path to template

        """
        #read input file and extract the fir row in a list
        with open(f"{ self.path }/template_pdbs.txt", "r") as infile:
            pdbs = infile.read().split(",")
        
        #remove last element of a list if it is empty
        if pdbs[-1] == "":
            pdbs.pop()
        print("READ_LIST: ", pdbs)
        
        if len(pdbs) > 1:
            self.shuffling_templates=True
        else:
            logging.warning("Impossible to shuffle with 1 template only.")
            
        return self.download_templates(pdbs)
