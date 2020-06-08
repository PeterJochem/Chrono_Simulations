# Summary
Juntao gave me the data he used to replicate Chen Li's graphs

# Results
The graphs look qualitaively correct. They seem to capture the rough structure of the original graphs. This is only a 2 layer neural network trained over 4000 epochs (about 3-5 minutes on cpu). The mapping struggles to get the scale correct though. I assume this is because our simulation's granular material is not exactly the same as the material used in Chen Li's experiment. This would explain the systematic error in the scales.
I experimented with a 5-6 layer network and training longer. This did not significantly improve the mapping. It took much much longer to train though (about 45 minutes). I think if I tune the hyper parameters more, the mapping could improve significantly.  
![Learned Mapping](https://github.com/PeterJochem/Chrono_Simulations/blob/master/replicateChenLiGraphs/Figure_1.png "Learned Mapping")

Here are the original graphs
![Original Graphs]( https://github.com/PeterJochem/Chrono_Simulations/blob/master/replicateChenLiGraphs/originalChenLiGraphs.png "Original Graphs")
