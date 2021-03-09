# Build AirSim with FST Driverless Environment on Ubuntu 18.04

## Install Unreal Engine

1. Download and install Unreal Engine 4.24.3 from this [link](https://docs.unrealengine.com/en-US/SharingAndReleasing/Linux/BeginnerLinuxDeveloper/SettingUpAnUnrealWorkflow/index.html). While the Unreal Engine is open source and free to download, registration is still required.

   **Note**: This project was tested only with UE 4.24.3. Other versions might work also.
   
## Build AirSim

  1. Download and build from this [link](https://microsoft.github.io/AirSim/build_linux/) AirSim.

## Creating and Setting Up Unreal Environment

You will need an Unreal project that hosts the environment. Follow the list below to create an environment that simulates the FSD competitions.
1. Make sure AirSim is built and Unreal Engine 4.24.3 is installed as described above.
2. Open UE editor and choose `New Project`. Choose `Game` then `Blank` with no starter content. Select your project's location, define it's name (`ProjectName` for example) and press `Create Project`.

![Screenshot from 2021-03-06 14-06-44](https://user-images.githubusercontent.com/38940464/110211450-566fcf00-7e9f-11eb-9188-8834359fe615.png)

3. After the project is loaded to the editor, from the `File menu` select `New C++ class`, leave default `None` on the type of class, click `Next`, leave default name `MyClass`, and click `Create Class`. We need to do this because Unreal requires at least one source file in project. It should trigger compile.

4. Close the UE editor.

5. Go to your folder for AirSim repo and copy `Unreal\Plugins` folder into your `ProjectName` folder. This way now your own Unreal project has AirSim plugin.

6. Download the environment assets of FSD racecourse from [here](https://drive.google.com/file/d/1lpBHRYYw7GRICgLaSfMQcXlbP2A98b9L/view?usp=sharing). Extract the zip into `ProjectName\Content`.

7. Download the formula Technion car assets from [here](https://drive.google.com/file/d/1PpR7k5hLZk5Gho--NwsvZ0OjMllWL0Qy/view?usp=sharing). Extract the zip into `ProjectName\Plugins\AirSim\Content\VehicleAdv\SUV` and select `replace` when asked for `SuvCarPawn.uasset` (the original file will be saved into a backup folder).

8. Edit the `ProjectName.uproject` so that it looks like this:
Notice that we called the project `ProjectName` in case you need to change it.

```
{
	"FileVersion": 3,
	"EngineAssociation": ### HERE SHOULD BE YOUR LICENSE, LEAVE IT AS IS ###,
	"Category": "Samples",
	"Description": "",
	"Modules": [
		{
			"Name": "ProjectName",
			"Type": "Runtime",
			"LoadingPhase": "Default",
			"AdditionalDependencies": [
				"AirSim"
			]
		}
	],
	"TargetPlatforms": [
		"MacNoEditor",
		"WindowsNoEditor"
	],
	"Plugins": [
		{
			"Name": "AirSim",
			"Enabled": true
		}
	]
}
```




9. Press `Enter` key on `ProjectName.uproject`. This will start the Unreal Editor. The Unreal Editor allows you to edit the environment, assets and other game related settings. 

10. First thing, load a map to set your environment. The maps are under `Content\RaceCourse\Maps`. To choose one of the maps double-click on it.

11. In `Window/World Settings` as shown below, set the `GameMode Override` to `AirSimGameMode`:

![Screenshot from 2021-03-06 17-25-51](https://user-images.githubusercontent.com/38940464/110211833-0f82d900-7ea1-11eb-81e6-4ad427a4fa53.png)

12. Next, if you want to change the location of `PlayerStart` object in your environment(`PlayerStart` object already exist) you can find and fix it in the `World Outliner`. This is where AirSim plugin will create and place the vehicle. If its too high up then vehicle will fall down as soon as you press play giving potentially random behavior.

![Screenshot from 2021-01-18 09-26-32](https://user-images.githubusercontent.com/38940464/110211872-40630e00-7ea1-11eb-9db6-b1898c3547b9.png)

13. Go to 'Edit->Editor Preferences' in Unreal Editor, in the 'Search' box type 'CPU' and ensure that the 'Use Less CPU when in Background' is unchecked. If you don't do this then UE will be slowed down dramatically when UE window loses focus.

14. Be sure to `Save` these edits. 

15. Hit the Play button in the Unreal Editor. See [how to use AirSim](https://github.com/Microsoft/AirSim/#how-to-use-it).


Ready... Set... GO!!!
You are now running AirSim in your FSD Unreal environment.

## Setup API control

You can use [APIs](https://github.com/FSTDriverless/AirSim/blob/master/docs/apis.md) for programmatic control or use the so-called [Computer Vision mode](https://github.com/FSTDriverless/AirSim/blob/master/docs/image_apis.md) to move around using the keyboard.
