import h5py
import signal
import jinja2 as j2
import json
from datetime import datetime
from collections import defaultdict
import sys
import time
import subprocess

__version__ = "0.5.3"


class SimulationError(Exception):
    """Custom error class for Simulation."""

    pass

class Simulation(list):
    def __init__(self, name="pylion"):
        super().__init__()

        # keep track of uids for list function overrides
        self._uids = []

        # slugify 'name' to use for filename
        name = name.replace(" ", "_").lower()

        self.attrs = {}
        self.attrs['gpu'] = None
        self.attrs['executable'] = 'lmp_serial'
        self.attrs['thermo_styles'] = ['step', 'cpu']
        self.attrs['timestep'] = 1e-6
        self.attrs['domain'] = [1e-3, 1e-3, 1e-3]  # length, width, height
        self.attrs['name'] = name
        self.attrs['neighbour'] = {'skin': 1, 'list': 'nsq'}
        self.attrs['coulombcutoff'] = 10
        self.attrs['template'] = 'simulation.j2'
        self.attrs['version'] = __version__
        self.attrs['rigid'] = {'exists': False}

    def __contains__(self, this):
        """Check if an item exists in the simulation using its ``uid``."""

        try:
            return this["uid"] in self._uids
        except KeyError:
            print("Item does not have a 'uid' key.")

    def append(self, this):
        """Appends the items and checks their attributes.
        Their ``uid`` is logged if they have one.
        """

        # only allow for dicts in the list
        if not isinstance(this, dict):
            raise SimulationError("Only 'dicts' are allowed in Simulation().")

        self._uids.append(this.get("uid"))

        # ions will always be included first so to sort you have
        # to give 1-count 'priority' keys to the rest
        if this.get("type") == "ions":
            this["priority"] = 0
            if this.get("rigid"):
                self.attrs["rigid"]["exists"] = True
                self.attrs["rigid"].setdefault("groups", []).append(this["uid"])

        timestep = this.get("timestep", 1e12)
        if timestep < self.attrs["timestep"]:
            self.attrs["timestep"] = timestep

        super().append(this)

    def extend(self, iterable):
        """Calls ``append`` on an iterable."""

        for item in iterable:
            self.append(item)

    def index(self, this):
        """Returns the index of an item using its ``uid``."""

        return self._uids.index(this["uid"])

    def remove(self, this):
        """Will not remove anything from the simulation but rather from lammps.
        It adds an ``unfix`` command when it's called.
        Use del if you really want to delete something or better yet don't
        add it to the simulation in the first place.
        """

        code = ["\n# Deleting a fix", f"unfix {this['uid']}\n"]
        self.append({"code": code, "type": "command"})

    def sort(self):
        """Sort with 'priority' keys if found otherwise do nothing."""

        try:
            super().sort(key=lambda item: item["priority"])
        except KeyError:
            pass
            # Not all elements have 'priority' keys. Cannot sort list

    def _writeinputfile(self):

        self.sort()  # if 'priority' keys exist

        odict = defaultdict(list)
        # deal the items in odict
        for item in self:
            if item.get("type") == "ions":
                odict["species"].append(item)
            else:
                odict["simulation"].append(item)

        # do a couple of checks
        # check for uids clashing
        uids = list(filter(None.__ne__, self._uids))
        if len(uids) > len(set(uids)):
            raise SimulationError(
                "There are identical 'uids'. Although this is allowed in some "
                " cases, 'lammps' is probably not going to like it."
            )

        # make sure species will behave
        maxuid = max(odict["species"], key=lambda item: item["uid"])["uid"]
        if maxuid > len(odict["species"]):
            raise SimulationError(
                f"Max 'uid' of species={maxuid} is larger than the number "
                f"of species={len(odict['species'])}. "
                "Calling '@lammps.ions' decorated functions increments the "
                "'uid' count unless it is for the same ion group."
            )

        # load jinja2 template
        env = j2.Environment(
            loader=j2.PackageLoader("pylion", "templates"), trim_blocks=True
        )
        template = env.get_template(self.attrs["template"])
        rendered = template.render({**self.attrs, **odict})

        with open(self.attrs["name"] + ".lammps", "w") as f:
            f.write(rendered)

        # get a few more attrs now that the lammps file is written
        # - simulation time
        self.attrs["time"] = datetime.now().isoformat()

        # - names of the output files
        fixes = filter(lambda item: item.get("type") == "fix", odict["simulation"])
        self.attrs["output_files"] = [
            line.split()[5]
            for fix in fixes
            for line in fix["code"]
            if line.startswith("dump")
        ]

    def execute(self):
        """Write lammps input file and run the simulation."""

        if getattr(self, "_hasexecuted", False):
            raise SimulationError(
                "Simulation has executed already. Do not run it again."
            )

        self._writeinputfile()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        cmd = self.attrs["executable"].split() + [
            "-log", self.attrs['name'] + ".lmp.log",
            "-in", self.attrs["name"] + ".lammps",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
        )
        for line in self.process.stdout:
            print(line, end='')
        self._hasexecuted = True
        return self.process.returncode
    # The user is responsible to attach this to their signal handlers,
    # recommended: https://stackoverflow.com/a/72592788/4935114
    def signal_handler(self, *args):
        if hasattr(self, 'process') and getattr(self, '_hasexecuted', False):
            self.process.send_signal(sig=signal.SIGTERM)
            self._hasexecuted = True
