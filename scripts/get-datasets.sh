# Copyright (C) 2024 Adrien ARNAUD
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/bash

baseUrl=http://www.doc.ic.ac.uk/~ahanda/

# Get ICL-living-room0
room0Archive=living_room_traj0_frei_png.tar.gz
 
mkdir -p data/ICL-living-room0/
mkdir -p tmp
wget ${baseUrl}/${room0Archive}
echo 'Unpacking archive...'
tar -xf ${room0Archive} -C tmp
echo 'Generating dataset...'
python3 scripts/dataset/ICL-living-room0/gen_ICL_living-room0.py
 
# Clean tmp data
rm -f ${room0Archive}
echo 'Done for ICL-living-room0'

## Get ICL-living-room1
room1Archive=living_room_traj1_frei_png.tar.gz

mkdir -p data/ICL-living-room1/
mkdir -p tmp
wget ${baseUrl}/${room1Archive}
echo 'Unpacking archive...'
tar -xf ${room1Archive} -C tmp
echo 'Generating dataset...'
python3 scripts/dataset/ICL-living-room1/gen_ICL_living-room1.py

# Clean tmp data
rm -rfd tmp
rm -f ${room1Archive}
echo 'Done for ICL-living-room1'

## Get ICL-living-room2
room2Archive=living_room_traj2_frei_png.tar.gz

mkdir -p data/ICL-living-room2/
mkdir -p tmp
wget ${baseUrl}/${room2Archive}
echo 'Unpacking archive...'
tar -xf ${room2Archive} -C tmp
echo 'Generating dataset...'
python3 scripts/dataset/ICL-living-room2/gen_ICL_living-room2.py

# Clean tmp data
rm -rfd tmp
rm -f ${room2Archive}
echo 'Done for ICL-living-room2'