for i in {1..10}
do
   # echo "Welcome $i times"
   let "z=10*$i"
   echo $z
   ./serial/dnn 123 2 $z $z 1 0.01 0.01 "./serial/a1a.test" "./serial/a1a.train" >> serial.out
   ./openCl/dnn 123 2 $z $z 1 0.01 0.01 "./openCl/a1a.test" "./openCl/a1a.train" >> openCl.out
done

for i in {2..5}
do
   # echo "Welcome $i times"
   let "z=100*$i"
   echo $z
   ./serial/dnn 123 2 $z $z 1 0.01 0.01 "./serial/a1a.test" "./serial/a1a.train" >> serial.out
   ./openCl/dnn 123 2 $z $z 1 0.01 0.01 "./openCl/a1a.test" "./openCl/a1a.train" >> openCl.out
done
