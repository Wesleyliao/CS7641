#!/bin/bash


# Declare arrays
declare -a linuxPackages=("python3-pip=9.0.1-2.3~ubuntu1"
                          "python3-tk=3.6.7-1~18.04"
                          "--user virtualenv")
declare -a pythonModules=("gym==0.12.1"
                          "numpy==1.15.1"
                          "scipy==1.1.0"
                          "scikit-learn==0.19.2"
                          "pandas==0.23.4"
                          "xlrd==0.9.0"
                          "matplotlib==2.2.3"
                          "seaborn==0.9.0"
                          "scikit-optimize==0.5.2")

# Initialize Constants
PROJECT_NAME="project3"
SECONDS=0
RED=`tput setaf 1`
GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
BLUE=`tput setaf 4`
BOLD=`tput bold`
NORMAL=`tput sgr0`


print_intro()
{
    # Print intro message
    echo -e "\n\n${BOLD}${BLUE}Setting up ML Project 1 environment...${NORMAL}\n\n\n"
}


setup_virtualenv()
{

    echo -e "\n\n${BOLD}${BLUE}Setting up Python virtual environment...${NORMAL}\n"

    # Create virtual environment
    python3 -m virtualenv .
    if [[ $? != 0 ]]; then
        echo -e "\n\n${BOLD}${RED}ERROR: failed to setup virtual environment $PROJECT_NAME${NORMAL}"
        echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
        exit
    fi

    # Activate virtual environment
    source bin/activate
    if [[ $? != 0 ]]; then
        echo -e "\n\n${BOLD}${RED}ERROR: failed to source virtualenv environment variables $PROJECT_NAME${NORMAL}"
        echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
        exit
    fi

    # Install required Python packages to virtual environment
    pip3 install -r requirements.txt
    if [[ $? != 0 ]]; then
        echo -e "\n\n${BOLD}${RED}ERROR: failed to install required Python packages $PROJECT_NAME${NORMAL}"
        echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
        exit
    fi
    echo -e "\n\n${BOLD}${BLUE}Python virtual environment setup complete${NORMAL}\n"

}


install_list()
{

    declare -a validInstallMethods=("apt" "pip3")

    # Ensure at least 2 arguments received
    let numVals=$#-1
    if [[ numVals < 2 ]]; then
        echo -e "\n\n${BOLD}${RED}ERROR: install_list() called with too few arguments: $#${NORMAL}"
        echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
        exit
    fi

    # Validate install method and construct install command from $1
    method=$1
    methodValid=false
    installCommandStart=""
    for i in "${validInstallMethods[@]}"; do
        if [[ "$method" == "$i" ]]; then
            methodValid=true
            installCommandStart="$method install --quiet"
            if [[ "$method" == "apt" ]]; then
                installCommandStart+="=2"
            fi
            if [[ "$method" == "pip3" ]]; then
                installCommandStart+=" --upgrade"
            fi
            break
        fi
    done
    if [[ $methodValid = false ]]; then
        echo ""
        echo -e "\n\n${BOLD}${RED}ERROR: install_list() called with invalid install method: $1${NORMAL}"
        echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
        exit
    fi

    # Loop through all arguments except first argument
    for i in "${@:2}"; do
        installCommand="$installCommandStart $i"
        if [[ "$method" == "pip3" ]]; then
            rc=0
            echo "Uninstalling all versions of ${BOLD}${BLUE}${i%%=*}${NORMAL}"
            while [[ "$rc" == 0 ]]; do
                # Uninstall previous versions
                pip3 uninstall --yes "${i%%=*}" 2>/dev/null
                rc=$?
                if [[ "$rc" == 1 ]]; then
                    pip3 uninstall --yes "${i%%=*}" 2>/dev/null
                    rc=$?
                fi
            done
        fi
        echo "Installing ${BOLD}${BLUE}$i${NORMAL} using ${BOLD}$method${NORMAL}"
        echo "Command: ${BOLD}${YELLOW}$installCommand${NORMAL}"
        stty -echo
        $installCommand
        stty echo
        if [[ $? != 0 ]]; then
            # Install failed
            echo -e "\n\n${BOLD}${RED}$i${NORMAL} install ${BOLD}${RED}FAILED${NORMAL}"
            echo -e "\n\n${BOLD}${RED}Exiting${NORMAL}\n\n"
            exit
        fi
        # Install successful
        echo "${BOLD}${BLUE}$i${NORMAL} install ${BOLD}${GREEN}SUCCESSFUL${NORMAL}"
        echo -e "\n"
        shift
    done

}


print_success()
{
    # Print success message
    echo -e "\n${BOLD}${BLUE}ML Project 1 environment setup complete${NORMAL}\n"
    echo -e "\t- Use ${BOLD}${BLUE}'source bin/activate'${NORMAL} to start the virtual environment"
    echo -e "\t- Execute ${BOLD}${BLUE}'python run_experiments.py --all --plot 2>&1 | tee output.log'${NORMAL} from within the virtual environment"
    echo -e "\t- Use ${BOLD}${BLUE}'deactivate'${NORMAL} to exit the virtual environment when done\n\n"
}


# Main
print_intro
install_list "apt" "${linuxPackages[@]}"
setup_virtualenv
print_success

