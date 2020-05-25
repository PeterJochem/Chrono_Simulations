#!/bin/bash
echo "Starting Data Generation"

# Remove the old data files
rm sim_data/output_plate_positions.csv
rm sim_data/output_plate_forces.csv

# ./myexe demo_GRAN_plate.json (intrusion angle) (attack angle)
for ((i = 0 ; i < 5; i++)); do
  for ((j = 0 ; j < 5; j++)); do
	  ./myexe demo_GRAN_plate.json $(echo "((3.14/2.0 - ($i * 3.14 / 4.0) ))" | bc -l) $(echo "((3.14/2.0 - ($j * 3.14 / 4.0) ))" | bc -l)
  done
done

echo "Finished Generating Data"
