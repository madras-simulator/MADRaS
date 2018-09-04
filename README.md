<h3 align="left"><img  width="100" height="100" src="Docs/img/logo_transparent.png"></h3>


# Description
MADRaS is a Multi-Agent Autonomous Driving Simulator built on top of TORCS. The simulator can be used to test autonomous vehicle algorithms both heuristic and learning based on an inherently multi agent setting.
# Installation prerequisities 

- Download [torcs-1.3.6](https://sourceforge.net/projects/torcs/files/all-in-one/1.3.6/torcs-1.3.6.tar.bz2/download), [plib-1.8.5](http://plib.sourceforge.net/download.html)

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

tar xvjf torcs-1.3.6.tar.bz2  
cd torcs-1.3.6/
./configure
make
sudo make install
sudo make datainstall
```
- test if torcs is running by typing `torcs` in a new terminal window
- Installing the scr-patch
  - Download [scr-patch](https://sourceforge.net/projects/cig/files/SCR%20Championship/Server%20Linux/2.1/scr-linux-patch.tgz/download)
  - unpack the scr at the base of the torcs repo `torcs-1.3.6/..`
 ``` shell
 cd scr-patch
 sh do_patch.sh #whenever prompted choose the default action [n]
 cd ..
 make
 sudo make install
 sudo make datainstall
 ```
 - test if scr client is installed or not.
   - open TORCS, navigate to configure race (race->quickrace->configure race -> select drivers) 
   - check the Not-Selected list for `scr-serverx` where x will range in [1,9]
- Installing the cpp scr client
  - Download [scr-client-cpp](https://sourceforge.net/projects/cig/files/SCR%20Championship/Client%20C%2B%2B/2.0/scr-client-cpp.tgz/download)
  - unpack the scr at the base of the torcs repo `torcs-1.3.6/..`
``` shell
cd scr-client-cpp
make
```
_Tested on ubuntu-16.04_
# Installation M.A.D.R.A.S

``` shell
git clone https://github.com/rudrasohan/M.A.D.R.A.S_dev.git
sudo -H pip3 install -r requirements.txt
```
### Maintainers
 - [Sohan Rudra](https://github.com/rudrasohan)
 - [Anirban Santara](https://github.com/Santara)
 - [Meha Kaushik](https://github.com/MehaKaushik)
