import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Users, Database, Zap, Network, Download, Upload, Brain, Shield, TrendingUp } from 'lucide-react';

const MLTorrent = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [currentEpoch, setCurrentEpoch] = useState(0);
    const [globalAccuracy, setGlobalAccuracy] = useState(0);
    const [peers, setPeers] = useState([]);
    const [networkActivity, setNetworkActivity] = useState([]);
    const [totalDataShared, setTotalDataShared] = useState(0);
    const [modelFragments, setModelFragments] = useState(0);
    const [trainingData, setTrainingData] = useState([]);
    const [globalModel, setGlobalModel] = useState(null);
    const intervalRef = useRef(null);

    // Generate synthetic training data (2D classification problem)
    const generateTrainingData = () => {
        const data = [];
        const numSamples = 1000;

        for (let i = 0; i < numSamples; i++) {
            const x1 = Math.random() * 10 - 5;
            const x2 = Math.random() * 10 - 5;
            // Create a simple decision boundary: y = 1 if x1 + x2 > 0, else 0
            const y = x1 + x2 + Math.random() * 0.5 - 0.25 > 0 ? 1 : 0;
            data.push({ x1, x2, y });
        }

        return data;
    };

    // Simple linear classifier
    class SimpleLinearClassifier {
        constructor() {
            this.weights = [Math.random() - 0.5, Math.random() - 0.5]; // w1, w2
            this.bias = Math.random() - 0.5;
        }

        predict(x1, x2) {
            const logit = this.weights[0] * x1 + this.weights[1] * x2 + this.bias;
            return 1 / (1 + Math.exp(-logit)); // sigmoid
        }

        train(data, learningRate = 0.01) {
            const gradients = [0, 0, 0]; // for w1, w2, bias

            for (const sample of data) {
                const prediction = this.predict(sample.x1, sample.x2);
                const error = sample.y - prediction;

                gradients[0] += error * sample.x1;
                gradients[1] += error * sample.x2;
                gradients[2] += error;
            }

            // Update weights
            this.weights[0] += learningRate * gradients[0] / data.length;
            this.weights[1] += learningRate * gradients[1] / data.length;
            this.bias += learningRate * gradients[2] / data.length;

            return gradients;
        }

        evaluate(data) {
            let correct = 0;
            for (const sample of data) {
                const prediction = this.predict(sample.x1, sample.x2) > 0.5 ? 1 : 0;
                if (prediction === sample.y) correct++;
            }
            return correct / data.length;
        }

        getWeights() {
            return {
                weights: [...this.weights],
                bias: this.bias
            };
        }

        setWeights(weights) {
            this.weights = [...weights.weights];
            this.bias = weights.bias;
        }
    }

    // Initialize peers with their own data shards and models
    useEffect(() => {
        const fullData = generateTrainingData();
        setTrainingData(fullData);

        // Create 5 peers with different data shards
        const peerCount = 5;
        const shardSize = Math.floor(fullData.length / peerCount);

        const initialPeers = [];
        for (let i = 0; i < peerCount; i++) {
            const start = i * shardSize;
            const end = i === peerCount - 1 ? fullData.length : start + shardSize;
            const peerData = fullData.slice(start, end);

            const peer = {
                id: i + 1,
                name: ['SmartHome-Berlin', 'IoT-Munich', 'EdgeDevice-Hamburg', 'SmartCity-Frankfurt', 'Home-Stuttgart'][i],
                model: new SimpleLinearClassifier(),
                data: peerData,
                accuracy: 0,
                status: 'active',
                contribution: 0.8 + Math.random() * 0.2,
                lastUpdate: Date.now(),
                downloadSpeed: 0,
                uploadSpeed: 0
            };

            // Initial accuracy calculation
            peer.accuracy = peer.model.evaluate(peer.data);
            initialPeers.push(peer);
        }

        setPeers(initialPeers);

        // Initialize global model
        const globalMod = new SimpleLinearClassifier();
        setGlobalModel(globalMod);
        setGlobalAccuracy(globalMod.evaluate(fullData));
    }, []);

    // Federated averaging function
    const federatedAveraging = (peerModels, contributions) => {
        const totalContribution = contributions.reduce((sum, c) => sum + c, 0);
        const avgWeights = [0, 0];
        let avgBias = 0;

        peerModels.forEach((model, i) => {
            const weight = contributions[i] / totalContribution;
            const modelWeights = model.getWeights();
            avgWeights[0] += modelWeights.weights[0] * weight;
            avgWeights[1] += modelWeights.weights[1] * weight;
            avgBias += modelWeights.bias * weight;
        });

        return {
            weights: avgWeights,
            bias: avgBias
        };
    };

    const performTrainingRound = () => {
        setPeers(prevPeers => {
            const updatedPeers = [...prevPeers];

            // Each peer trains on its local data
            updatedPeers.forEach(peer => {
                if (peer.status === 'active') {
                    // Local training
                    peer.model.train(peer.data);
                    peer.accuracy = peer.model.evaluate(peer.data);

                    // Simulate network activity
                    peer.uploadSpeed = Math.random() * 100 + 50;
                    peer.downloadSpeed = Math.random() * 100 + 50;
                    peer.lastUpdate = Date.now();
                }
            });

            // Federated averaging (global model update)
            const activeModels = updatedPeers.filter(p => p.status === 'active').map(p => p.model);
            const contributions = updatedPeers.filter(p => p.status === 'active').map(p => p.contribution);

            if (activeModels.length > 0) {
                const avgWeights = federatedAveraging(activeModels, contributions);
                globalModel.setWeights(avgWeights);

                // Update global accuracy
                const newGlobalAccuracy = globalModel.evaluate(trainingData);
                setGlobalAccuracy(newGlobalAccuracy);

                // Distribute updated model back to peers (gossip protocol simulation)
                updatedPeers.forEach(peer => {
                    if (peer.status === 'active' && Math.random() > 0.3) {
                        // Peer receives partial global model update
                        const currentWeights = peer.model.getWeights();
                        const globalWeights = globalModel.getWeights();

                        // Blend local and global models
                        const blendFactor = 0.3;
                        peer.model.setWeights({
                            weights: [
                                currentWeights.weights[0] * (1 - blendFactor) + globalWeights.weights[0] * blendFactor,
                                currentWeights.weights[1] * (1 - blendFactor) + globalWeights.weights[1] * blendFactor
                            ],
                            bias: currentWeights.bias * (1 - blendFactor) + globalWeights.bias * blendFactor
                        });
                    }
                });
            }

            // Random peer status changes
            updatedPeers.forEach(peer => {
                if (Math.random() < 0.05) {
                    peer.status = peer.status === 'active' ? 'syncing' : 'active';
                }
            });

            return updatedPeers;
        });

        // Add network activity
        setNetworkActivity(prev => {
            const newActivities = [];
            const activePeers = peers.filter(p => p.status === 'active');

            // Generate 2-4 network activities per round
            for (let i = 0; i < Math.floor(Math.random() * 3) + 2; i++) {
                if (activePeers.length >= 2) {
                    const from = activePeers[Math.floor(Math.random() * activePeers.length)];
                    const to = activePeers[Math.floor(Math.random() * activePeers.length)];

                    if (from.id !== to.id) {
                        newActivities.push({
                            id: Date.now() + i,
                            from: from.id,
                            to: to.id,
                            type: Math.random() > 0.4 ? 'model' : 'gradient',
                            size: Math.floor(Math.random() * 150) + 50,
                            timestamp: Date.now()
                        });
                    }
                }
            }

            return [...prev.slice(-15), ...newActivities];
        });

        setTotalDataShared(prev => prev + Math.floor(Math.random() * 300) + 200);
        setModelFragments(prev => prev + Math.floor(Math.random() * 8) + 3);
    };

    const startTraining = () => {
        setIsTraining(true);
        intervalRef.current = setInterval(() => {
            setCurrentEpoch(prev => prev + 1);
            performTrainingRound();
        }, 1500); // Training round every 1.5 seconds
    };

    const stopTraining = () => {
        setIsTraining(false);
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }
    };

    const resetTraining = () => {
        stopTraining();
        setCurrentEpoch(0);
        setGlobalAccuracy(0);
        setNetworkActivity([]);
        setTotalDataShared(0);
        setModelFragments(0);

        // Reinitialize everything
        const fullData = generateTrainingData();
        setTrainingData(fullData);

        const peerCount = 5;
        const shardSize = Math.floor(fullData.length / peerCount);

        const resetPeers = [];
        for (let i = 0; i < peerCount; i++) {
            const start = i * shardSize;
            const end = i === peerCount - 1 ? fullData.length : start + shardSize;
            const peerData = fullData.slice(start, end);

            const peer = {
                id: i + 1,
                name: ['SmartHome-Berlin', 'IoT-Munich', 'EdgeDevice-Hamburg', 'SmartCity-Frankfurt', 'Home-Stuttgart'][i],
                model: new SimpleLinearClassifier(),
                data: peerData,
                accuracy: 0,
                status: 'active',
                contribution: 0.8 + Math.random() * 0.2,
                lastUpdate: Date.now(),
                downloadSpeed: 0,
                uploadSpeed: 0
            };

            peer.accuracy = peer.model.evaluate(peer.data);
            resetPeers.push(peer);
        }

        setPeers(resetPeers);

        const globalMod = new SimpleLinearClassifier();
        setGlobalModel(globalMod);
        setGlobalAccuracy(globalMod.evaluate(fullData));
    };

    const getStatusColor = (status) => {
        switch(status) {
            case 'active': return 'text-green-400';
            case 'syncing': return 'text-yellow-400';
            case 'downloading': return 'text-blue-400';
            default: return 'text-gray-400';
        }
    };

    const getStatusIcon = (status) => {
        switch(status) {
            case 'active': return <Zap className="w-4 h-4" />;
            case 'syncing': return <RotateCcw className="w-4 h-4 animate-spin" />;
            case 'downloading': return <Download className="w-4 h-4" />;
            default: return <Users className="w-4 h-4" />;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                        MLTorrent
                    </h1>
                    <p className="text-xl text-gray-300 mb-6">
                        Distributed Machine Learning via P2P Networks
                    </p>
                    <div className="flex justify-center gap-4 mb-8">
                        <button
                            onClick={isTraining ? stopTraining : startTraining}
                            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                                isTraining
                                    ? 'bg-red-600 hover:bg-red-700'
                                    : 'bg-green-600 hover:bg-green-700'
                            }`}
                        >
                            {isTraining ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                            {isTraining ? 'Stop Training' : 'Start Training'}
                        </button>
                        <button
                            onClick={resetTraining}
                            className="flex items-center gap-2 px-6 py-3 rounded-lg font-semibold bg-gray-600 hover:bg-gray-700 transition-all"
                        >
                            <RotateCcw className="w-5 h-5" />
                            Reset
                        </button>
                    </div>
                </div>

                {/* Main Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Brain className="w-8 h-8 text-purple-400" />
                            <h3 className="text-lg font-semibold">Global Model</h3>
                        </div>
                        <div className="text-3xl font-bold text-purple-400">
                            {(globalAccuracy * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-gray-300">Accuracy</p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <TrendingUp className="w-8 h-8 text-green-400" />
                            <h3 className="text-lg font-semibold">Epoch</h3>
                        </div>
                        <div className="text-3xl font-bold text-green-400">
                            {currentEpoch}
                        </div>
                        <p className="text-sm text-gray-300">Training Rounds</p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Users className="w-8 h-8 text-blue-400" />
                            <h3 className="text-lg font-semibold">Active Peers</h3>
                        </div>
                        <div className="text-3xl font-bold text-blue-400">
                            {peers.filter(p => p.status === 'active').length}
                        </div>
                        <p className="text-sm text-gray-300">Connected</p>
                    </div>

                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <div className="flex items-center gap-3 mb-2">
                            <Database className="w-8 h-8 text-cyan-400" />
                            <h3 className="text-lg font-semibold">Data Shared</h3>
                        </div>
                        <div className="text-3xl font-bold text-cyan-400">
                            {(totalDataShared / 1000).toFixed(1)}K
                        </div>
                        <p className="text-sm text-gray-300">Model Updates</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Peer Network */}
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                            <Network className="w-6 h-6" />
                            Peer Network
                        </h3>
                        <div className="space-y-3">
                            {peers.map(peer => (
                                <div key={peer.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <div className={`${getStatusColor(peer.status)}`}>
                                            {getStatusIcon(peer.status)}
                                        </div>
                                        <div>
                                            <div className="font-semibold">{peer.name}</div>
                                            <div className="text-sm text-gray-400">
                                                {peer.data.length} samples | {(peer.accuracy * 100).toFixed(1)}% local acc
                                            </div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm text-gray-300">
                                            ↑{peer.uploadSpeed.toFixed(0)} ↓{peer.downloadSpeed.toFixed(0)} KB/s
                                        </div>
                                        <div className="text-xs text-gray-400">
                                            Trust: {(peer.contribution * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Network Activity */}
                    <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                            <Zap className="w-6 h-6" />
                            Network Activity
                        </h3>
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                            {networkActivity.slice(-10).map(activity => (
                                <div key={activity.id} className="flex items-center justify-between p-2 bg-white/5 rounded text-sm">
                                    <div className="flex items-center gap-2">
                                        {activity.type === 'model' ?
                                            <Brain className="w-4 h-4 text-purple-400" /> :
                                            <TrendingUp className="w-4 h-4 text-green-400" />
                                        }
                                        <span>
                      Peer {activity.from} → Peer {activity.to}
                    </span>
                                    </div>
                                    <div className="flex items-center gap-2 text-gray-300">
                                        <span>{activity.size}KB</span>
                                        {activity.type === 'model' ?
                                            <Upload className="w-3 h-3" /> :
                                            <Download className="w-3 h-3" />
                                        }
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Training Info */}
                <div className="mt-6 bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
                    <h3 className="text-lg font-semibold mb-3">Training Details</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                            <span className="text-gray-300">Dataset:</span> {trainingData.length} samples (2D classification)
                        </div>
                        <div>
                            <span className="text-gray-300">Algorithm:</span> Federated Averaging with Gossip Protocol
                        </div>
                        <div>
                            <span className="text-gray-300">Model:</span> Linear Classifier with Sigmoid Activation
                        </div>
                    </div>
                </div>

                {/* Footer Info */}
                <div className="mt-8 text-center">
                    <div className="flex items-center justify-center gap-6 text-sm text-gray-400">
            <span className="flex items-center gap-1">
              <Shield className="w-4 h-4" />
              Federated Learning
            </span>
                        <span className="flex items-center gap-1">
              <Network className="w-4 h-4" />
              P2P Protocol
            </span>
                        <span className="flex items-center gap-1">
              <Database className="w-4 h-4" />
                            {modelFragments} Model Fragments
            </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MLTorrent;