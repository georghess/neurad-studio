# Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import subprocess
import sys

CURRENT_YEAR = str(datetime.datetime.now().year)
COPYRIGHT_STR = f"# Copyright {CURRENT_YEAR} the authors of NeuRAD and contributors.\n"
LICENSE_HEADER = """\
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


def count_copyrights(lines):
    count = 0
    for line in lines:
        if line.startswith("# Copyright"):
            count += 1
        else:
            break
    return count


def main():
    files = sys.argv[1:]
    for file_name in files:
        if file_name.startswith("tests"):  # exclude tests directory
            continue
        with open(file_name, "r") as f:
            lines = f.readlines()

            if lines[0] != COPYRIGHT_STR:
                git_status = subprocess.run(
                    ["git", "status", "--porcelain", "--", file_name],
                    check=True,
                    encoding="utf-8",
                    stdout=subprocess.PIPE,
                ).stdout[0]
                if git_status == "A" and lines[0].startswith("# Copyright"):
                    lines[0] = COPYRIGHT_STR
                else:
                    lines.insert(0, COPYRIGHT_STR)

            if LICENSE_HEADER not in "".join(lines):
                lines.insert(count_copyrights(lines), LICENSE_HEADER)

        with open(file_name, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
