<h3 align="left"><img  width="100" height="100" src="Docs/img/logo_transparent.png"></h3>


# Description
MADRaS is a Multi-Agent Autonomous Driving Simulator built on top of TORCS. The simulator can be used to test autonomous vehicle algorithms both heuristic and learning based on an inherently multi agent setting.

Note please : The repository is under active developement and re-design. Currently the `master` branch has the Single-Agent version of MADRaS whereas the Multi-Agent part is in the `Version-2` branch. 

## Installation
### Installation prerequisities 
- TORCS
```shell
git clone https://github.com/madras-simulator/TORCS.git
```
- [plib](http://plib.sourceforge.net/)
- Install Dependencies
``` shell

sudo apt-get install libalut-dev 
sudo apt-get install libvorbis-dev 
sudo apt-get install libxrandr2 libxrandr-dev 
sudo apt-get install zlib1g-dev 
sudo apt-get install libpng-dev 
sudo apt-get install libplib-dev libplib1 
sudo apt-get install python-tk
sudo apt-get install xautomation
```
- Installling plib (follow instructions on the plib page)
- Installing TORCS
``` shell 
cd TORCS/
./configure --prefix=$HOME/usr/local
make && make install
make datainstall
export PATH=$HOME/usr/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/usr/local/lib:$LD_LIBRARY_PATH
```
- test if torcs is running by typing `torcs` in a new terminal window
- test if scr client is installed or not.
  - open TORCS, navigate to configure race (race->quickrace->configure race -> select drivers) 
  - check the Not-Selected list for `scr-serverx` where x will range in [1,9]

_Tested on ubuntu-16.04 & ubuntu-18.04_

### Installation MADRaS

``` shell
# if req an env can also be created
git clone https://github.com/madras-simulator/MADRaS
cd MADRaS/
pip3 install -e .
```

For further information regarding the simulator please checkout our [Wiki](https://github.com/madras-simulator/MADRaS/wiki)
 
## Maintainers
 - [Sohan Rudra](https://github.com/rudrasohan)
 - [Anirban Santara](https://github.com/Santara)
 - [Meha Kaushik](https://github.com/MehaKaushik)
 
 ## Credits
 
 ### Developers:
 - [Abhishek Naik](https://github.com/abhisheknaik96)
 - [Sohan Rudra](https://github.com/rudrasohan)
 - [Meha Kaushik](https://github.com/MehaKaushik)
 - [Buridi Aditya](https://github.com/buridiaditya)
 
 ### Project Manager:
 - [Anirban Santara](https://github.com/Santara)
 
 ### Mentors:
 - [Bharat Kaul](https://ai.intel.com/bio/bharat-kaul/)
 - [Balaraman Ravindran](https://www.cse.iitm.ac.in/~ravi/) 
