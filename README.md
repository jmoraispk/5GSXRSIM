# SRXSIM

This radio simulator oriented for virtual reality meetings is fully described in the paper: 
[Performance Modelling and Assessment for Social VR Conference Applications in 5G Radio Networks](https://ieeexplore.ieee.org/document/9606263)
by João Morais, Sjors Braam, Remco Litjens, Sandra Kizhakkekundil and Hans van Den Berg.

It uses the [Quadriga](https://quadriga-channel-model.de/) stochastic channel model to simulate the wireless channels in a meeting room where XR meetings happen. Then, one can configure almost 100 parameters to evaluate their impact on the packet drop rate during VR meetings. These parameters span 3 layers: the physical layer (e.g., number of antennas), the MAC layer (e.g., scheduler and beam management parameters), and the application layer (e.g., frame and video compression parameters). 

## Dependencies

1. Windows (for the parallelization tool)
2. Python Libraries:
    * numpy
    * scipy
    * joblib     (          for generation) 
    * matplotlib (          for data analysis)
    * moviepy    (optional, for data analysis. To install do: conda install -c conda-forge moviepy) 
    * PyPDF2     (optional, for data analysis. To install do: conda install -c conda-forge pypdf2) 

3. Matlab Runtime Compiler 9.8 (for Matlab 2020a)


## Suggested Development Environment

[Spyder](https://www.spyder-ide.org/), installed through Anaconda.

For installing:
1. Install anaconda
2. Open anaconda prompt
3. Update anaconda: conda update conda
4. Search the most recent version of spyder: conda search -f spyder
5. Install that version (say 5.0.5 is currently the most recent): conda install spyder=5.0.5 

For updating to a new version of Spyder, repeat steps 2 to 5.


# Works that used this simulator

1. The 1st research publication:
[Assessment of 5G RAN Features for Integrated Services Provisioning in Smart Cities
](https://ieeexplore.ieee.org/abstract/document/9941612) by Ayushi Kandoi, Maria Raftopoulou and Remco Litjens

2. The 1st MSc thesis:
[Analysis of radio downlink in Social XR scenarios involving 5G channel state information acquisition techniques](https://repository.tudelft.nl/islandora/object/uuid%3A56795e15-bc0f-4578-a5b8-3cae0d98101d?collection=education) by Sandra Kizhakkekundil (advised by Remco Litjens and Alexander Yarovoy)

3. The 2nd MSc thesis and research publication: 
[Application-level performance of cross-layer scheduling for social VR in 5G
](https://ieeexplore.ieee.org/document/10118957) by Zheng Du, Hans van Den Berg, Toni Dimitrovski and Remco Litjens

4. The 3rd MSc thesis:
[Offloading Strategies for the Rendering Process in Edge-enabled Virtual Reality Games](https://essay.utwente.nl/93300/) by Cathy Schmit
