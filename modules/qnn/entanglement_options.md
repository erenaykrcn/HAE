'full' entanglement is each qubit is entangled with all the others. <br>

'linear' entanglement is qubit  entangled with qubit , for all , where  is the total number of qubits. <br>

'reverse_linear' entanglement is qubit  entangled with qubit , for all , where  is the total number of qubits. Note that if entanglement_blocks = 'cx' then this option provides the same unitary as 'full' with fewer entangling gates. <br>

'pairwise' entanglement is one layer where qubit  is entangled with qubit , for all even values of , and then a second layer where qubit  is entangled with qubit , for all odd values of . <br>

'circular' entanglement is linear entanglement but with an additional entanglement of the first and last qubit before the linear part. <br>

'sca' (shifted-circular-alternating) entanglement is a generalized and modified version of the proposed circuit 14 in Sim et al.. It consists of circular entanglement where the ‘long’ entanglement connecting the first with the last qubit is shifted by one each block. Furthermore the role of control and target qubits are swapped every block (therefore alternating).