#!/usr/bin/env python3
import serial
import time

if __name__ == '__main__':
	ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
	ser.flush()

	first_command = "MV00X10Y03Z06Q00"
	second_command = "MV00X00Y00Z00Q00"
	
	ser.write(bytes(first_command,"utf-8"))
	print("first push")
	#time.sleep(30)
	
	#print("second push")
	#ser.write(bytes(second_command,"utf-8"))
	#time.sleep(30)
	while 1 :	
		line = ser.readline().decode('utf-8').rstrip()
		print(line)
		if(line =="Z Done"):
			ser.write(bytes(second_command,"utf-8"))
			print("second push")
			
