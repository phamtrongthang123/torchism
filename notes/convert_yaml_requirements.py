import yaml

with open("f1.yml") as file_handle:
    environment_data = yaml.load(file_handle,  Loader=yaml.FullLoader)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        print(dependency)
        package_name, package_version,_ = dependency.split("=")
        file_handle.write("{}=={}\n".format(package_name, package_version))
