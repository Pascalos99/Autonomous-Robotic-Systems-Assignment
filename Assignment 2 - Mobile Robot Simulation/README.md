# Controls
With `Q & A` the speed of the left wheel can be controlled. `Q` increases speed, `A` decreases speed. <br>
With `E & D` the speed of the right wheel can be controlled. `E` increases speed, `D` decreases speed. <br>
With `W & S` the speed of both weels can be controlled. `W` increases speed, `S` decreases speed. <br>
With `X` the speed of both weels can be reset to zero. <br>

# Configuration
The config.ini file contains the configurable variables. The variables under the "BOT" section are pretty straight forward, <br>
`radius` defines the size of the bot this can be a decimal number, `num_sensors` defines the amount of sensors on the bot, <br>
`vision_range` defines how far the sensors can "see" and `direction` is the direction the bot is facing when starting the game. <br>

The variables under the "PROGRAM" section define the presentatation of the program. <br>
`speed_step` controls how much speed is added to a wheel with each button press.
`sensor_data_separate` controls how the sensor data is presented. It has two options `0` or `1`: <br>
* Option `0` will show the sensor distance on the sensor lines. <br>
* Option `1` will show the sensor distance as a seperate list on the right-hand side, <br>
  on the sensor lines a number will be presented which corresponds to the number in the list.
