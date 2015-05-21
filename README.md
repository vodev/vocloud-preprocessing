Data Preprocessing Worker for VO-cloud
######################################

:author: Andrej Palicka
:license: MIT


The application works as a plugin of vo-cloud data mining application. The main purpose is to
transform FITS files into generaly readable format - CSV.

It can perform many transformations which are described by //config.json//

'classes_file' -- file with classes assigned to (FITS) data files in form of JSON as

```
 {
  "<fits-file-without-extension>": <class>,
  "<fits-file-without-extension>": <class>,
  "<fits-file-without-extension>": <class>
 }
```

'normalize' -- normalize values (in the second column) between 0 an 1.0

'binning' -- bins the intenzities based on their difference between the followings

'remove_duplicates' --

'select_features' -- in json_dict:
