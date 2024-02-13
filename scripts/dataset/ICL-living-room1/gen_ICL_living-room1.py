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

import os

if __name__ == '__main__':
    i = 0
    with open('scripts/dataset/ICL-living-room1/dataset.txt', 'r') as fp:
        for l in fp:
            depth, color, tx, ty, tz, qx, qy, qz, qw = l.rstrip('\n').split(' ')
            os.system('cp tmp/{} data/ICL-living-room1/depth_{}.png'.format(depth, i))
            os.system('cp tmp/{} data/ICL-living-room1/color_{}.png'.format(color, i))
            os.system('echo {} {} {} {} {} {} {} > data/ICL-living-room1/pose_{}.txt'.\
                      format(tx, ty, tz, qx, qy, qz, qw, i))
            i += 1
        os.system('cp scripts/dataset/ICL-living-room1/intrinsics.txt data/ICL-living-room1/intrinsics.txt')
